"""Frozen diagnostic controller: a continuously advancing teacher takes control."""
from dataclasses import dataclass
import hashlib
import json

from soku_rl.policy.base import Policy


def validate_takeover_frame(after_frames):
    if type(after_frames) is not int or after_frames < 0:
        raise ValueError("teacher takeover requires nonnegative integer after_frames")


@dataclass(frozen=True)
class TeacherTakeoverPolicy(Policy):
    name: str
    learner: object
    teacher: object
    after_frames: int

    def __post_init__(self):
        validate_takeover_frame(self.after_frames)

    @property
    def fingerprint(self):
        identity = ["teacher-takeover-v1", self.learner.fingerprint,
                    self.teacher.fingerprint, self.after_frames]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def spawn(self, seed):
        # Preserve each constituent's reference seed, with separate actor memory
        # and private RNG. No extra sampling may perturb the learner prefix.
        return self.spawn_with_seeds(seed, seed)

    def spawn_with_seeds(self, learner_seed, teacher_seed):
        return TeacherTakeoverActor(self.learner.spawn(learner_seed), self.teacher.spawn(teacher_seed),
                                    self.after_frames, 0)


@dataclass
class TeacherTakeoverActor:
    learner: object
    teacher: object
    after_frames: int
    frame: int

    def act(self, observation):
        return self.act_labelled(observation)[1]

    def act_labelled(self, observation):
        """Advance once, returning the teacher label and actually executed action."""
        teacher_action = self.teacher.act(observation)
        action = self.learner.act(observation) if self.frame < self.after_frames else teacher_action
        self.frame += 1
        return teacher_action, action
