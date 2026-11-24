
import os, time, datetime, pandas as pd, torch

DRIVE  = "/content/drive/MyDrive/PROYEK UAS DEEP LEARNING"
LOGDIR = f"{DRIVE}/logs"; os.makedirs(LOGDIR, exist_ok=True)

class RunLog:
    """Mencatat satu run ke log_<owner>.csv, termasuk run yang gagal dan durasi GPU."""
    def __init__(self, owner, **config):
        ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.row = dict(run_id=f"{owner}_{ts}", owner=owner, **config)

    def __enter__(self):
        self.t0 = time.time()
        self.row["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return self

    def metrics(self, **m):
        self.row.update(m)

    def __exit__(self, et, ev, tb):
        self.row["gpu_minutes"] = round((time.time() - self.t0) / 60, 2)
        self.row["status"] = "failed" if et else "ok"
        if et: self.row["notes"] = f"{et.__name__}: {str(ev)[:150]}"
        path = f"{LOGDIR}/log_{self.row['owner']}.csv"
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        pd.concat([old, pd.DataFrame([self.row])]).to_csv(path, index=False)
        return False
