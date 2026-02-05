use crate::pipeline::frames::Frame;
use crate::pipeline::processor::Processor;
use anyhow::Result;
use tokio::sync::mpsc;

#[allow(dead_code)]
pub struct Pipeline {
    processors: Vec<Box<dyn Processor>>,
    tx: mpsc::Sender<Frame>,
    rx: mpsc::Receiver<Frame>,
}

#[allow(dead_code)]
impl Pipeline {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(100);
        Self {
            processors: Vec::new(),
            tx,
            rx,
        }
    }

    pub fn add_processor(&mut self, processor: Box<dyn Processor>) {
        self.processors.push(processor);
    }

    pub async fn run(mut self) -> Result<()> {
        while let Some(frame) = self.rx.recv().await {
            let mut frames_to_process = vec![frame];
            
            for processor in &mut self.processors {
                let mut next_frames = Vec::new();
                for f in frames_to_process {
                    let processed = processor.process(f).await?;
                    next_frames.extend(processed);
                }
                frames_to_process = next_frames;
                if frames_to_process.is_empty() {
                    break;
                }
            }
        }
        Ok(())
    }

    pub fn get_tx(&self) -> mpsc::Sender<Frame> {
        self.tx.clone()
    }
}
