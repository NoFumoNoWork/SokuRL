"""Rollout-local accounting of earned rewards, including unfinished episodes."""
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


PARTS=('damage','combo','counter','passive','terminal')
ENDS=('win','loss','timeout','draw')


class RewardBatch:
    def __init__(self, scale, start_steps):
        if not math.isfinite(scale) or scale<=0:raise ValueError('invalid reward scale')
        self.scale,self.start_steps=scale,start_steps
        self.transitions=0
        self.hp={key:0. for key in (*PARTS,*ENDS)}
        self.counts={key:0 for key in ENDS}
        self.reward_sum=0.

    def add(self, parts, reward, end=None):
        if set(parts)!=set(PARTS) or end not in (*ENDS,None):
            raise ValueError('invalid reward batch schema')
        if not all(math.isfinite(v) for v in (*parts.values(),reward)):
            raise ValueError('nonfinite reward component')
        if end is None and parts['terminal']!=0:
            raise ValueError('terminal reward without terminal outcome')
        self.transitions+=1
        self.reward_sum+=float(reward)
        for key in PARTS:self.hp[key]+=float(parts[key])
        if end:
            self.counts[end]+=1
            self.hp[end]+=float(parts['terminal'])

    def row(self, index, steps, complete):
        if steps-self.start_steps!=self.transitions:
            raise ValueError('reward coverage does not match learner transitions')
        normalized={key:value/self.scale for key,value in self.hp.items()}
        return dict(schema=1,batch=index,start_steps=self.start_steps,end_steps=steps,
                    transitions=self.transitions,complete=complete,scale=self.scale,
                    outcome_counts=self.counts.copy(),hp=self.hp.copy(),normalized=normalized,
                    normalized_per_transition={key:value/self.transitions for key,value in normalized.items()},
                    learner_reward_sum=self.reward_sum,
                    reward_residual=self.reward_sum-sum(normalized[key] for key in PARTS))


def append_batch(directory,row):
    directory=Path(directory)
    with (directory/'reward_batches.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
    flat={key:row[key] for key in ('batch','start_steps','end_steps','transitions','complete','scale')}
    for group in ('hp','normalized','normalized_per_transition','outcome_counts'):
        flat.update({f'{group}/{key}':value for key,value in row[group].items()})
    flat.update(learner_reward_sum=row['learner_reward_sum'],reward_residual=row['reward_residual'])
    for key,value in row.get('combo_stats',{}).items():
        if isinstance(value,(int,float)) or value is None:flat[f'combo_stats/{key}']=value
    for key in ('native_hit_length_histogram','independent_segment_lower_bound_histogram'):
        if key in row.get('combo_stats',{}):flat[f'combo_stats/{key}']=json.dumps(row['combo_stats'][key])
    path=directory/'reward_batches.csv'
    new=not path.exists()
    with path.open('a',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat))
        if new:writer.writeheader()
        writer.writerow(flat)


def plot_batches(directory):
    """Dependency-free SVG; one point is one sampling rollout, not an SGD epoch."""
    directory=Path(directory)
    rows=[json.loads(line) for line in (directory/'reward_batches.jsonl').read_text().splitlines()]
    if not rows:return
    groups=(('Terminal',('win','loss','timeout')),('Passive',('passive',)),('Combo reward',('combo',)))
    _plot_rows(rows,directory/'reward_batches.svg',groups,('HP sum','normalized / transition'),
               'Earned rewards per PPO sampling rollout (partial final batch retained)')
    combat=[]
    for row in rows:
        stats=row.get('combo_stats',{})
        observed=stats.get('available_transitions',0)
        left={key:stats.get(key) if observed else None for key in
              ('hits','combo_extensions','counter_openers','counter_taken','unknown_counter_hits')}
        right={key:value*1000/observed if value is not None else None for key,value in left.items()}
        left.update(native_length=stats.get('mean_completed_hits'),
                    segments_lower_bound=stats.get('mean_completed_segments'))
        right.update(native_length=stats.get('max_observed_hits') if observed else None,
                     segments_lower_bound=stats.get('max_observed_segments') if observed else None)
        combat.append(dict(end_steps=row['end_steps'],hp=left,normalized_per_transition=right))
    groups=(('Hits / extensions',('hits','combo_extensions')),
            ('Counter Hit',('counter_openers','counter_taken','unknown_counter_hits')),
            ('Combo length',('native_length','segments_lower_bound')))
    _plot_rows(combat,directory/'combo_batches.svg',groups,
               ('counts; length: completed mean','per 1k observed steps; length: observed max'),
               'Native combo telemetry: damaging hits and independent action segments are distinct')


def _plot_rows(rows,destination,groups,unit_names,title):
    ns='http://www.w3.org/2000/svg'
    ET.register_namespace('',ns)
    root=ET.Element(f'{{{ns}}}svg',dict(width='1100',height='800',viewBox='0 0 1100 800'))
    def node(tag,**attributes):return ET.SubElement(root,f'{{{ns}}}{tag}',{k:str(v) for k,v in attributes.items()})
    def label(x,y,value,size=12):
        node('text',x=x,y=y,fill='#202124',**{'font-family':'Arial,sans-serif','font-size':size}).text=value
    node('rect',x=0,y=0,width=1100,height=800,fill='white')
    label(30,28,title,17)
    colors=('#16834b','#c53939','#9c6500')
    for row_index,(title,keys) in enumerate(groups):
        for column,unit in enumerate(('hp','normalized_per_transition')):
            left,top=75+column*545,82+row_index*235
            width,height=440,160
            label(left,top-16,title+' / '+unit_names[column],13)
            values=[r[unit][key] for r in rows for key in keys if r[unit][key] is not None]
            low,high=min([0.,*values]),max([0.,*values])
            if low==high:low,high=(0.,1.) if values else (-1.,1.)
            margin=(high-low)*.08
            if low<0:low-=margin
            if high>0:high+=margin
            first,last=rows[0]['end_steps'],rows[-1]['end_steps']
            x=lambda step:left+(step-first)/max(1,last-first)*width
            y=lambda value:top+height-(value-low)/(high-low)*height
            for v in (low,(low+high)/2,high):
                node('line',x1=left,y1=y(v),x2=left+width,y2=y(v),stroke='#dddddd')
                label(left-65,y(v)+4,f'{v:.3g}',11)
            node('line',x1=left,y1=y(0),x2=left+width,y2=y(0),stroke='#777777')
            for index,key in enumerate(keys):
                color='#c53939' if key=='passive' else colors[index]
                segments=[[]]
                for r in rows:
                    if r[unit][key] is None:
                        if segments[-1]:segments.append([])
                        continue
                    segments[-1].append(f'{x(r["end_steps"]):.2f},{y(r[unit][key]):.2f}')
                    node('circle',cx=x(r['end_steps']),cy=y(r[unit][key]),r=2,fill=color)
                for points in segments:
                    if points:node('polyline',points=' '.join(points),fill='none',stroke=color,**{'stroke-width':2})
                label(left+index*130,top+height+37,key)
            if not values:label(left+50,top+height/2,'No confirmed observations in this range')
            label(left,top+height+18,str(first))
            label(left+width-55,top+height+18,str(last))
    label(75,787,'X: learner transitions. Gaps mean unavailable observations or no completed combo, not zero length.')
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)
