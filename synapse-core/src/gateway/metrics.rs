/// Observability module for Synapse.
///
/// Provides structured logging and metrics collection.
/// - Structured logging via `tracing`
/// - Prometheus-compatible metrics endpoint at `/metrics`
/// - Tracks jobs, tasks, latency, and errors
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Metrics collector for tracking job and task statistics.
///
/// Thread-safe counters for concurrent access.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    inner: Arc<MetricsInner>,
}

#[derive(Debug)]
struct MetricsInner {
    /// Total jobs submitted
    jobs_submitted: AtomicU64,
    /// Total jobs completed
    jobs_completed: AtomicU64,
    /// Total jobs failed
    jobs_failed: AtomicU64,
    /// Total tasks dispatched
    tasks_dispatched: AtomicU64,
    /// Total tasks completed
    tasks_completed: AtomicU64,
    /// Total tasks failed
    tasks_failed: AtomicU64,
    /// Total retries
    task_retries: AtomicU64,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    /// Create a new metrics collector with zero counters.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                jobs_submitted: AtomicU64::new(0),
                jobs_completed: AtomicU64::new(0),
                jobs_failed: AtomicU64::new(0),
                tasks_dispatched: AtomicU64::new(0),
                tasks_completed: AtomicU64::new(0),
                tasks_failed: AtomicU64::new(0),
                task_retries: AtomicU64::new(0),
            }),
        }
    }

    /// Record a job submission.
    pub fn record_job_submitted(&self) {
        self.inner.jobs_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a job completion.
    pub fn record_job_completed(&self) {
        self.inner.jobs_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a job failure.
    pub fn record_job_failed(&self) {
        self.inner.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a task dispatch.
    pub fn record_task_dispatched(&self) {
        self.inner.tasks_dispatched.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a task completion.
    pub fn record_task_completed(&self) {
        self.inner.tasks_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a task failure.
    pub fn record_task_failed(&self) {
        self.inner.tasks_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a task retry.
    pub fn record_task_retry(&self) {
        self.inner.task_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current metrics as a snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            jobs_submitted: self.inner.jobs_submitted.load(Ordering::Relaxed),
            jobs_completed: self.inner.jobs_completed.load(Ordering::Relaxed),
            jobs_failed: self.inner.jobs_failed.load(Ordering::Relaxed),
            tasks_dispatched: self.inner.tasks_dispatched.load(Ordering::Relaxed),
            tasks_completed: self.inner.tasks_completed.load(Ordering::Relaxed),
            tasks_failed: self.inner.tasks_failed.load(Ordering::Relaxed),
            task_retries: self.inner.task_retries.load(Ordering::Relaxed),
        }
    }

    /// Export metrics in Prometheus text format.
    pub fn export_prometheus(&self) -> String {
        let snap = self.snapshot();
        format!(
            "# HELP synapse_jobs_submitted Total jobs submitted\n\
             # TYPE synapse_jobs_submitted counter\n\
             synapse_jobs_submitted {}\n\
             # HELP synapse_jobs_completed Total jobs completed\n\
             # TYPE synapse_jobs_completed counter\n\
             synapse_jobs_completed {}\n\
             # HELP synapse_jobs_failed Total jobs failed\n\
             # TYPE synapse_jobs_failed counter\n\
             synapse_jobs_failed {}\n\
             # HELP synapse_tasks_dispatched Total tasks dispatched\n\
             # TYPE synapse_tasks_dispatched counter\n\
             synapse_tasks_dispatched {}\n\
             # HELP synapse_tasks_completed Total tasks completed\n\
             # TYPE synapse_tasks_completed counter\n\
             synapse_tasks_completed {}\n\
             # HELP synapse_tasks_failed Total tasks failed\n\
             # TYPE synapse_tasks_failed counter\n\
             synapse_tasks_failed {}\n\
             # HELP synapse_task_retries Total task retries\n\
             # TYPE synapse_task_retries counter\n\
             synapse_task_retries {}\n",
            snap.jobs_submitted,
            snap.jobs_completed,
            snap.jobs_failed,
            snap.tasks_dispatched,
            snap.tasks_completed,
            snap.tasks_failed,
            snap.task_retries,
        )
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsSnapshot {
    pub jobs_submitted: u64,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub tasks_dispatched: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub task_retries: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_collector_starts_at_zero() {
        let collector = MetricsCollector::new();
        let snap = collector.snapshot();
        assert_eq!(snap.jobs_submitted, 0);
        assert_eq!(snap.jobs_completed, 0);
        assert_eq!(snap.jobs_failed, 0);
        assert_eq!(snap.tasks_dispatched, 0);
        assert_eq!(snap.tasks_completed, 0);
        assert_eq!(snap.tasks_failed, 0);
        assert_eq!(snap.task_retries, 0);
    }

    #[test]
    fn metrics_collector_records_job_submitted() {
        let collector = MetricsCollector::new();
        collector.record_job_submitted();
        collector.record_job_submitted();
        assert_eq!(collector.snapshot().jobs_submitted, 2);
    }

    #[test]
    fn metrics_collector_records_job_completed() {
        let collector = MetricsCollector::new();
        collector.record_job_submitted();
        collector.record_job_completed();
        assert_eq!(collector.snapshot().jobs_completed, 1);
    }

    #[test]
    fn metrics_collector_records_job_failed() {
        let collector = MetricsCollector::new();
        collector.record_job_submitted();
        collector.record_job_failed();
        assert_eq!(collector.snapshot().jobs_failed, 1);
    }

    #[test]
    fn metrics_collector_records_task_dispatched() {
        let collector = MetricsCollector::new();
        collector.record_task_dispatched();
        assert_eq!(collector.snapshot().tasks_dispatched, 1);
    }

    #[test]
    fn metrics_collector_records_task_completed() {
        let collector = MetricsCollector::new();
        collector.record_task_dispatched();
        collector.record_task_completed();
        assert_eq!(collector.snapshot().tasks_completed, 1);
    }

    #[test]
    fn metrics_collector_records_task_failed() {
        let collector = MetricsCollector::new();
        collector.record_task_dispatched();
        collector.record_task_failed();
        assert_eq!(collector.snapshot().tasks_failed, 1);
    }

    #[test]
    fn metrics_collector_records_task_retry() {
        let collector = MetricsCollector::new();
        collector.record_task_retry();
        assert_eq!(collector.snapshot().task_retries, 1);
    }

    #[test]
    fn metrics_collector_clone_shares_state() {
        let collector = MetricsCollector::new();
        let cloned = collector.clone();
        collector.record_job_submitted();
        assert_eq!(cloned.snapshot().jobs_submitted, 1);
    }

    #[test]
    fn metrics_export_prometheus_format() {
        let collector = MetricsCollector::new();
        collector.record_job_submitted();
        collector.record_job_completed();
        collector.record_task_dispatched();

        let prometheus = collector.export_prometheus();
        assert!(prometheus.contains("synapse_jobs_submitted 1"));
        assert!(prometheus.contains("synapse_jobs_completed 1"));
        assert!(prometheus.contains("synapse_tasks_dispatched 1"));
        assert!(prometheus.contains("# HELP"));
        assert!(prometheus.contains("# TYPE synapse_jobs_submitted counter"));
    }

    #[test]
    fn metrics_snapshot_equality() {
        let snap1 = MetricsSnapshot {
            jobs_submitted: 1,
            jobs_completed: 2,
            jobs_failed: 3,
            tasks_dispatched: 4,
            tasks_completed: 5,
            tasks_failed: 6,
            task_retries: 7,
        };
        let snap2 = snap1.clone();
        assert_eq!(snap1, snap2);
    }
}
