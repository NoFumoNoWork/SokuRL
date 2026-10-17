"""Bounded god-script or trusted checkpoint versus original TH123 CPU evaluation."""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from sokurl_platform import sokurl
from sokurl_platform.bridge_shared import BridgeClient, wait_for_steps, RawFrameState
from sokurl_platform.frames import wait_for_frame_zero
from sokurl_platform.combat_capture import CombatCaptureClient
from sokurl_platform.memory import ProcessMemory
from game_runtime.privileged import PrivilegedReader
from soku_rl.env.encoding import decode_action
from soku_rl.policy.god.package import ScriptPackage
from soku_rl.policy.god.runtime import GodPolicy
from soku_rl.env.hisouten_env import EpisodeConfig
from soku_rl.env.observation.visibility import VisibilityConfig
from soku_rl.env.match import LEGACY_MATCH

ROOT = Path(os.environ['SOKURL_RL_ROOT']).parent


class CheckpointControls:
    """Reuse the saved policy's exact animation/history/action contracts."""
    def __init__(self, training_directory, checkpoint, device):
        from omegaconf import OmegaConf
        import torch
        from soku_rl.env.wrappers.learning import LearningConfig, LearningInterface
        from soku_rl.policy.loader import load_policy
        self.directory = training_directory.resolve(strict=True)
        self.path = (self.directory/checkpoint).resolve(strict=True)
        if not self.path.is_relative_to(self.directory):
            raise ValueError('checkpoint must belong to training directory')
        config = OmegaConf.to_container(OmegaConf.load(self.directory/'config.yaml'),resolve=True)
        self.episode = EpisodeConfig.from_dict(config['episode'])
        if (self.episode.observation_mode!='animation' or self.episode.decision_frames!=1
                or self.episode.latency_frames!=0 or
                tuple(p.character for p in (self.episode.match.player_0,self.episode.match.player_1))!=(1,0)):
            raise ValueError('CPU evaluation requires saved animation/1F/zero-latency Marisa-left/Reimu-right contract')
        torch.set_num_threads(config['rl']['cpu_threads'])
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        self.interface = LearningInterface(self.episode,LearningConfig(**config['wrappers']))
        self.policy = load_policy('historical-selected-checkpoint',dict(kind='sb3',path=str(self.path),
            training_config=str(self.directory/'config.yaml')),self.interface,device)
        self.metadata = dict(path=str(self.path),sha256=self.policy.fingerprint,
            config_sha256=hashlib.sha256((self.directory/'config.yaml').read_bytes()).hexdigest(),
            model_steps=int(self.policy.model.num_timesteps),device=str(device),
            inference='existing PPOEpisode seeded categorical sampling; not greedy',
            observation_shape=list(self.interface.observation_space.shape),
            action_count=int(self.interface.action_space.n),horizon=self.episode.max_frames)

    def reset(self):
        from soku_rl.env.observation_history import ObservationHistory
        from soku_rl.env.wrappers.learning import LearningEpisode
        self.history = ObservationHistory(self.episode,('player_0','player_1'))
        self.learning = LearningEpisode(self.interface)

    def observation(self, raw, render):
        from soku_rl.env.observation.animation import observe_animation
        observations=observe_animation(raw,render,self.episode.visibility,
            self.episode.animation_object_slots,self.episode.max_frames)
        frame=int(raw.frameId)
        if frame==0:
            self.history.reset(frame,observations)
            self.learning.reset_agent('player_0',self.history.observations()['player_0'])
        else:
            self.history.append(frame,observations)
        return self.learning.observation('player_0',self.history.observations()['player_0'],frame)

    def command(self, action):
        command=self.interface.command(action)
        self.learning.record_command('player_0',command)
        return decode_action(command).inputs


def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def play(args):
    if not 1 <= args.games <= 10 or not 1 <= args.max_frames <= 18000:
        raise ValueError('bounded games/frame horizon required')
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('preserve previous evidence')
    out.mkdir(parents=True)
    checkpoint = (CheckpointControls(args.training_directory,args.checkpoint,args.device)
                  if args.training_directory else None)
    if checkpoint and args.smoke:
        raise ValueError('checkpoint evaluation cannot use neutral smoke mode')
    package = ScriptPackage(ROOT/'third_party/th123_ai/package/th123ai/script',
                            ROOT/'third_party/th123_ai/source/th123_ai/api.ai')
    # Existing GodPolicy only uses these contract fields in this direct reader path.
    from types import SimpleNamespace
    policy = GodPolicy('full-marisa-neutral-home', package, '01_marisa_main.ai',
        SimpleNamespace(observation_mode='privileged_state',decision_frames=1,latency_frames=0))
    if checkpoint:
        policy=checkpoint.policy
        args.max_frames=min(args.max_frames,checkpoint.episode.max_frames)
    candidate='checkpoint' if checkpoint else 'god'
    protocol = dict(games=args.games, max_frames=args.max_frames, script=getattr(policy,'script',None),
        package_sha256=package.fingerprint, policy_fingerprint=policy.fingerprint,
        god_character=1,cpu_character=0,god_seat=0,cpu_seat=1,stage=0,music=0,
        god_noise=0,god_extra_delays=0,weather='clear',cards_disabled=True,
        cpu_difficulty='INHERITED_ORIGINAL_CONFIG_NOT_YET_DECODED',
        game_config_sha256=hashlib.sha256((ROOT/'th123_jp/config123.dat').read_bytes()).hexdigest(),
        reset='native_scene_reload_no_snapshot_no_fresh_reuse',
        result_definition='single_KO_episode_not_best_of_three; timeout reported separately',
        mode='smoke' if args.smoke else '10_game_evaluation',
        cpu_input='no_external_override; native null keyManager',
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'platform/native/SokuRLBridge/SokuRLBridge.cpp',
            'platform/src/sokurl_platform/native_reader.py',
            'platform/src/sokurl_platform/sokurl.py', 'rl/tools/native_cpu_benchmark.py')})
    protocol['candidate']=candidate
    if checkpoint:
        protocol['checkpoint']=checkpoint.metadata
        protocol['package_sha256']=None
    save(out/'protocol.json',protocol)
    report = dict(status='RUNNING',games=[],protocol=protocol)
    started=time.perf_counter()
    try:
        with ExitStack() as resources:
            # Existing environment switches are scoped to this process tree.
            from unittest.mock import patch
            resources.enter_context(patch.dict(os.environ,{
                'SOKURL_FRESH_RESET':'0','SOKURL_NATIVE_INPUT_COUNTERS':'1',
                'SOKURL_COMBAT_EVENTS':'1','SOKURL_NATIVE_TRACE':'0'}))
            processes=sokurl._launch_vs_group_from_title(1,60.,headless=True,unlimited=True,
                seed=918042743,pause_at_start=True,characters=(1,0),cards_disabled=True,
                native_cpu=True,stage=0,music=0,capture_state=bool(checkpoint))
            process=processes[0]
            def close_game():
                if process.is_running():
                    sokurl.shutdown(5.,process.pid)
            resources.callback(close_game)
            client=resources.enter_context(BridgeClient(process.pid))
            raw=wait_for_frame_zero(client,process.pid,60.)
            save(out/'initial-native.json',dict(mode=int(raw.battleMode),
                characters=[int(raw.p1.characterId),int(raw.p2.characterId)],stage=int(raw.stageId),
                hp=[int(raw.p1.hp),int(raw.p2.hp)],seed=int(raw.randomSeed)))
            (out/'initial.raw').write_bytes(bytes(raw))
            memory=ProcessMemory(process.pid); resources.callback(memory.close)
            reader=PrivilegedReader(memory)
            combat=CombatCaptureClient(process.pid); resources.callback(combat.close)
            image=None
            if checkpoint:
                from sokurl_platform.images import ImageClient
                image=ImageClient(process.pid);resources.callback(image.close)
            for game in range(args.games):
                seed=918042743+game*104729
                if game:
                    sequence=client.reset_episode(seed)
                    raw=client.wait_for_reset(sequence,int(raw.segmentId)+1,seed,60.)
                combat.drain()
                actor=policy.spawn(728341+game)
                if checkpoint: checkpoint.reset()
                actions=[Counter(),Counter()];damage=[0,0];hits=[0,0]
                ch=[0,0];cpu_nonzero=0;objects=[0,0];hitstop_frames=0
                decisions=Counter(); command_count=0;dead=False
                entry=dict(game=game+1,seed=seed,script_seed=728341+game,pid=process.pid,
                           status='RUNNING',start_hp=[int(raw.p1.hp),int(raw.p2.hp)])
                if entry['start_hp'] != [10000,10000]:
                    raise RuntimeError('not a fresh full-HP episode')
                frame=0
                match_started=time.perf_counter()
                with gzip.open(out/f'game-{game+1:02d}.frames.gz','wb') as trace:
                    while frame < args.max_frames:
                        snapshot=client.snapshot();raw=snapshot.latest
                        if (raw.battleMode!=2 or raw.sceneId!=5 or snapshot.game_frame!=frame
                                or snapshot.run_state_name!='PAUSED'):
                            raise RuntimeError('original CPU frame boundary/mode mismatch')
                        if (raw.p1.characterId,raw.p2.characterId)!=(1,0):
                            raise RuntimeError('character mismatch')
                        if frame and (raw.activeWeather!=21 or raw.displayedWeather!=21):
                            raise RuntimeError('Clear not enforced')
                        if raw.stageId!=0 or raw.p1.cardCount or raw.p2.cardCount:
                            raise RuntimeError('non-neutral stage or native cards present')
                        if frame==0:
                            sequence=client.set_fixed_weather('clear')
                            ack=client.wait_for_ack(sequence,10.)
                            if ack.result_name not in ('COMPLETE','CHECKPOINT_INVALIDATED'):
                                raise RuntimeError('weather command failed')
                        observed=reader.observe(raw,client)[0]
                        if observed.world['stage_number']==10 or observed.world['bgm_number'] in (11,41):
                            raise RuntimeError('Marisa home advantage condition active')
                        if any(observed.players[s]['deck'] for s in (0,1)):
                            raise RuntimeError('CPU no-card deck invariant failed')
                        if frame==0:
                            entry['world']=observed.world
                            entry['initial_skill_levels']=[p['skills'] for p in observed.players]
                        cpu_nonzero+=int(any(observed.players[1]['keys']))
                        for seat,player in enumerate((raw.p1,raw.p2)):
                            actions[seat][int(player.actionId)]+=1
                        objects[0]+=int(raw.p1ObjectCount>0);objects[1]+=int(raw.p2ObjectCount>0)
                        hitstop_frames+=int(bool(raw.p1.hitstop or raw.p2.hitstop))
                        trace.write(bytes(raw))
                        if raw.p1.hp<=0 or raw.p2.hp<=0:
                            dead=True;break
                        if args.smoke:
                            # Neutral learner must leave the opponent genuinely autonomous.
                            command=(0,0,0,0,0,0,0,0)
                        else:
                            if checkpoint:
                                encoded=checkpoint.observation(raw,image.read_state(frame,10.))
                                command=checkpoint.command(actor.act(encoded))
                            else:
                                command=decode_action(actor.act_observation(observed)).inputs
                        if command[6] or command[7]:
                            if checkpoint:
                                raise RuntimeError('saved checkpoint requested card input; preserve action contract')
                            command=(*command[:6],0,0)
                        decisions[str(command)]+=1
                        sequence=client.step_with_inputs(command,(0,)*8)
                        command_count+=1
                        raw=wait_for_steps([client],[sequence],[frame+1],10.)[0].latest
                        client.drain_frames()
                        frame+=1
                        for e in combat.drain():
                            if e['frame']!=frame or e['segment']!=raw.segmentId:
                                raise RuntimeError('native combat event frame mismatch')
                            if e['kind']=='HIT_RESOLUTION' and e['hit_or_block']=='HIT':
                                seat=e['attacker_seat']
                                damage[seat]+=max(0,e['hp_before']-e['hp_after'])
                                hits[seat]+=int(e['hp_before']>e['hp_after'])
                                ch[seat]+=int(e.get('counter_hit_first') is True)
                            if e['kind']=='ACTION_TRANSITION' and e.get('accepted') and e['action_after']>=600:
                                raise RuntimeError('card-only action accepted during no-card test')
                    raw=client.snapshot().latest
                    # Include an endpoint reached exactly at the horizon.
                    dead=raw.p1.hp<=0 or raw.p2.hp<=0
                winner=('draw' if raw.p1.hp<=0 and raw.p2.hp<=0 else
                        'native_cpu' if raw.p1.hp<=0 else candidate if raw.p2.hp<=0 else 'timeout')
                if getattr(actor,'failures',[]):raise RuntimeError('community script failure')
                if client.snapshot().dropped_frames:raise RuntimeError('dropped native frames')
                if any(p.actionId>=600 for p in (raw.p1,raw.p2)):
                    raise RuntimeError('card-only endpoint action')
                if not any(a>=300 for a in actions[1]):
                    raise RuntimeError('native CPU never attacked')
                entry.update(status='PASS',winner=winner,frames=frame,seconds=time.perf_counter()-match_started,
                    final_hp=[int(raw.p1.hp),int(raw.p2.hp)],damage=damage,hits=hits,counter_hits=ch,
                    native_action_frames=[dict(a) for a in actions],cpu_nonzero_key_frames=cpu_nonzero,
                    projectile_frames=objects,hitstop_frames=hitstop_frames,external_commands=command_count,
                    script_failures=getattr(actor,'failures',[]),
                    script_logs=[str(x) for x in getattr(actor,'logs',[])[:10]],
                    candidate_requested_inputs=dict(decisions),dropped_frames=client.snapshot().dropped_frames)
                if not checkpoint:
                    entry['god_requested_inputs']=dict(decisions)
                report['games'].append(entry)
                save(out/'result.json',report)
                print(json.dumps({k:v for k,v in entry.items() if k in
                    ('game','winner','frames','seconds','final_hp','hits','cpu_nonzero_key_frames') }),flush=True)
            totals=Counter(g['winner'] for g in report['games'])
            report.update(status='PASS',outcomes=dict(totals),
                candidate_win_rate=totals[candidate]/len(report['games']),
                limits=['CPU difficulty inherited, not claimed highest.',
                        'Left-god/right-CPU only, no seat symmetry claim.',
                        '10 single-KO episodes; Clear and no cards; not general strength ranking.'])
            if not checkpoint:
                report['god_win_rate']=report['candidate_win_rate']
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        report['seconds']=time.perf_counter()-started
        save(out/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('games','protocol')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--games',type=int,default=10)
    parser.add_argument('--max-frames',type=int,default=18000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--training-directory',type=Path)
    parser.add_argument('--checkpoint',default='final.zip')
    parser.add_argument('--device',default='cpu')
    play(parser.parse_args())
