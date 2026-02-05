use crate::pipeline::frames::Frame;
use anyhow::Result;
use async_trait::async_trait;

#[allow(dead_code)]
#[async_trait]
pub trait Processor: Send + Sync {
    async fn process(&mut self, frame: Frame) -> Result<Vec<Frame>>;
}
