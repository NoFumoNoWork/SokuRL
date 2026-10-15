"""Keep interactive games alive while a slow worker finishes its current request."""
from soku_rl.env.worker_pipe import WorkerConnection


class PlayConnection(WorkerConnection):
    def _read_response(self, operation):
        if operation in {"poll", "submit"}:
            # The pipe reader still reports EOF/worker failure. A delayed reply
            # is not a reason to close the games or send a duplicate request.
            return self.replies.get()
        return super()._read_response(operation)
