use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::Serialize;

/// Collects inference metrics for the scheduler.
///
/// All counters are atomic for thread-safe concurrent access.
/// Queue and execution times are collected in a mutex-protected vec
/// for percentile calculation.
pub struct MetricsCollector {
    total_jobs: AtomicU64,
    completed_jobs: AtomicU64,
    failed_jobs: AtomicU64,
    total_tasks: AtomicU64,
    retried_tasks: AtomicU64,
    queue_times_ms: Mutex<Vec<u64>>,
    execution_times_ms: Mutex<Vec<u64>>,
}

/// Snapshot of collected metrics.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsReport {
    /// Total number of jobs submitted.
    pub total_jobs: u64,
    /// Number of jobs that completed successfully.
    pub completed_jobs: u64,
    /// Number of jobs that failed permanently.
    pub failed_jobs: u64,
    /// Ratio of completed to total jobs (0.0–1.0).
    pub success_rate: f64,
    /// Total number of tasks dispatched.
    pub total_tasks: u64,
    /// Number of tasks that were retried at least once.
    pub retried_tasks: u64,
    /// Ratio of retried tasks to total tasks (0.0–1.0).
    pub retry_rate: f64,
    /// Queue time p50 in milliseconds.
    pub queue_time_p50_ms: u64,
    /// Queue time p95 in milliseconds.
    pub queue_time_p95_ms: u64,
    /// Queue time p99 in milliseconds.
    pub queue_time_p99_ms: u64,
    /// Execution time p50 in milliseconds.
    pub execution_time_p50_ms: u64,
    /// Execution time p95 in milliseconds.
    pub execution_time_p95_ms: u64,
    /// Execution time p99 in milliseconds.
    pub execution_time_p99_ms: u64,
}

impl MetricsCollector {
    /// Creates a new collector with all counters at zero.
    pub fn new() -> Self {
        Self {
            total_jobs: AtomicU64::new(0),
            completed_jobs: AtomicU64::new(0),
            failed_jobs: AtomicU64::new(0),
            total_tasks: AtomicU64::new(0),
            retried_tasks: AtomicU64::new(0),
            queue_times_ms: Mutex::new(Vec::new()),
            execution_times_ms: Mutex::new(Vec::new()),
        }
    }

    /// Records a job submission.
    pub fn record_job_submit(&self) {
        self.total_jobs.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a job completion.
    pub fn record_job_complete(&self) {
        self.completed_jobs.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a job failure.
    pub fn record_job_fail(&self) {
        self.failed_jobs.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a task dispatch with queue time and execution time in milliseconds.
    pub fn record_task_dispatch(&self, queue_ms: u64, exec_ms: u64) {
        self.total_tasks.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut times) = self.queue_times_ms.lock() {
            times.push(queue_ms);
        }
        if let Ok(mut times) = self.execution_times_ms.lock() {
            times.push(exec_ms);
        }
    }

    /// Records a task retry.
    pub fn record_task_retry(&self) {
        self.retried_tasks.fetch_add(1, Ordering::Relaxed);
    }

    /// Generates a snapshot of all collected metrics.
    ///
    /// Percentiles are calculated from the collected time samples.
    /// Returns zeros for percentiles when no samples exist.
    pub fn report(&self) -> MetricsReport {
        let total = self.total_jobs.load(Ordering::Relaxed);
        let completed = self.completed_jobs.load(Ordering::Relaxed);
        let failed = self.failed_jobs.load(Ordering::Relaxed);
        let tasks = self.total_tasks.load(Ordering::Relaxed);
        let retries = self.retried_tasks.load(Ordering::Relaxed);

        let success_rate = if total > 0 { completed as f64 / total as f64 } else { 0.0 };
        let retry_rate = if tasks > 0 { retries as f64 / tasks as f64 } else { 0.0 };

        let queue_times = self.queue_times_ms.lock().unwrap();
        let exec_times = self.execution_times_ms.lock().unwrap();

        MetricsReport {
            total_jobs: total,
            completed_jobs: completed,
            failed_jobs: failed,
            success_rate,
            total_tasks: tasks,
            retried_tasks: retries,
            retry_rate,
            queue_time_p50_ms: percentile(&queue_times, 50),
            queue_time_p95_ms: percentile(&queue_times, 95),
            queue_time_p99_ms: percentile(&queue_times, 99),
            execution_time_p50_ms: percentile(&exec_times, 50),
            execution_time_p95_ms: percentile(&exec_times, 95),
            execution_time_p99_ms: percentile(&exec_times, 99),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculates the p-th percentile from a slice of values.
///
/// Sorts a copy of the data internally. Returns 0 for empty slices.
fn percentile(data: &[u64], p: u64) -> u64 {
    if data.is_empty() {
        return 0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let idx = ((p as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report() {
        let mc = MetricsCollector::new();
        let r = mc.report();
        assert_eq!(r.total_jobs, 0);
        assert_eq!(r.success_rate, 0.0);
        assert_eq!(r.queue_time_p50_ms, 0);
    }

    #[test]
    fn record_and_report() {
        let mc = MetricsCollector::new();
        mc.record_job_submit();
        mc.record_job_submit();
        mc.record_job_complete();
        mc.record_job_fail();

        mc.record_task_dispatch(10, 100);
        mc.record_task_dispatch(20, 200);
        mc.record_task_retry();

        let r = mc.report();
        assert_eq!(r.total_jobs, 2);
        assert_eq!(r.completed_jobs, 1);
        assert_eq!(r.failed_jobs, 1);
        assert_eq!(r.success_rate, 0.5);
        assert_eq!(r.total_tasks, 2);
        assert_eq!(r.retried_tasks, 1);
        assert_eq!(r.retry_rate, 0.5);
        assert!(r.queue_time_p50_ms > 0);
        assert!(r.execution_time_p50_ms > 0);
    }

    #[test]
    fn percentile_values() {
        assert_eq!(percentile(&[], 50), 0);
        assert_eq!(percentile(&[100], 50), 100);
        assert_eq!(percentile(&[10, 20, 30, 40, 50], 50), 30);
        assert_eq!(percentile(&[10, 20, 30, 40, 50], 95), 50);
    }
}
