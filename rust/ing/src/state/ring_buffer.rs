//! Fixed-size ring buffer for time series data

use std::collections::VecDeque;

/// A fixed-size ring buffer that maintains the most recent N items
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a new item, removing the oldest if at capacity
    pub fn push(&mut self, item: T) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(item);
    }

    /// Get the number of items in the buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.data.len() >= self.capacity
    }

    /// Get the most recent item
    pub fn last(&self) -> Option<&T> {
        self.data.back()
    }

    /// Get the oldest item
    pub fn first(&self) -> Option<&T> {
        self.data.front()
    }

    /// Get item at index (0 = oldest)
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get the last N items as a slice (newest last)
    pub fn last_n(&self, n: usize) -> Vec<&T> {
        let start = self.data.len().saturating_sub(n);
        self.data.iter().skip(start).collect()
    }

    /// Iterate over all items (oldest first)
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Get all items as a vector
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Clone + Copy> RingBuffer<T> {
    /// Compute returns (current / previous - 1) for numeric types
    pub fn returns(&self) -> Vec<f64>
    where
        T: Into<f64>,
    {
        if self.data.len() < 2 {
            return Vec::new();
        }

        self.data
            .iter()
            .zip(self.data.iter().skip(1))
            .map(|(prev, curr)| {
                let prev_f: f64 = (*prev).into();
                let curr_f: f64 = (*curr).into();
                if prev_f != 0.0 {
                    curr_f / prev_f - 1.0
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl RingBuffer<f64> {
    /// Compute mean of all values
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Compute standard deviation
    pub fn std(&self) -> f64 {
        if self.data.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Compute min value
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Compute max value
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);

        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
        assert!(buf.is_full());

        buf.push(4);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.first(), Some(&2));
        assert_eq!(buf.last(), Some(&4));
    }

    #[test]
    fn test_ring_buffer_stats() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(5);

        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);
        buf.push(5.0);

        assert_eq!(buf.mean(), 3.0);
        assert!((buf.std() - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_ring_buffer_returns() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(5);

        buf.push(100.0);
        buf.push(110.0);
        buf.push(105.0);

        let returns = buf.returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10);  // 110/100 - 1
        assert!((returns[1] - (-0.04545454545454545)).abs() < 1e-10);  // 105/110 - 1
    }
}
