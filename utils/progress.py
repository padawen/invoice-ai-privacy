import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading

class ProgressTracker:
    """Thread-safe progress tracking for invoice processing jobs"""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 3600  # 1 hour

    def create_job(self, filename: str) -> str:
        """Create a new processing job and return job ID"""
        job_id = str(uuid.uuid4())

        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "filename": filename,
                "status": "started",
                "progress": 0,
                "stage": "initializing",
                "message": "Processing started",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "stages": {
                    "upload": {"progress": 0, "status": "pending", "duration": None},
                    "ocr": {"progress": 0, "status": "pending", "duration": None},
                    "llm": {"progress": 0, "status": "pending", "duration": None},
                    "postprocess": {"progress": 0, "status": "pending", "duration": None}
                },
                "error": None,
                "result": None
            }

        return job_id

    def update_progress(self, job_id: str, stage: str, progress: int, message: str = "", status: str = "processing"):
        """Update job progress"""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job["stage"] = stage
                job["progress"] = min(100, max(0, progress))
                job["message"] = message
                job["status"] = status
                job["updated_at"] = datetime.utcnow().isoformat()

                # Update specific stage
                if stage in job["stages"]:
                    job["stages"][stage]["status"] = status
                    job["stages"][stage]["progress"] = progress

    def update_stage_duration(self, job_id: str, stage: str, duration: float):
        """Update stage completion duration"""
        with self._lock:
            if job_id in self._jobs and stage in self._jobs[job_id]["stages"]:
                self._jobs[job_id]["stages"][stage]["duration"] = duration
                self._jobs[job_id]["stages"][stage]["status"] = "completed"

    def set_error(self, job_id: str, error: str):
        """Set job error status"""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = error
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job"""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job["status"] in ["started", "processing"]:
                    job["status"] = "cancelled"
                    job["message"] = "Processing cancelled by user"
                    job["updated_at"] = datetime.utcnow().isoformat()
                    return True
        return False

    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled"""
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]["status"] == "cancelled"
        return False

    def set_result(self, job_id: str, result: Dict[str, Any]):
        """Set job completion result"""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["progress"] = 100
                self._jobs[job_id]["result"] = result
                self._jobs[job_id]["message"] = "Processing completed successfully"
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current job progress"""
        with self._lock:
            return self._jobs.get(job_id, None)

    def cleanup_old_jobs(self):
        """Remove jobs older than cleanup interval"""
        cutoff = datetime.utcnow() - timedelta(seconds=self._cleanup_interval)

        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                created_at = datetime.fromisoformat(job["created_at"])
                if created_at < cutoff:
                    to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]

    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all current jobs (for debugging)"""
        with self._lock:
            return self._jobs.copy()

# Global progress tracker instance
progress_tracker = ProgressTracker()