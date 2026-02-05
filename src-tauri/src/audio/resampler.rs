use anyhow::Result;
use rubato::{FftFixedIn, Resampler as RubatoResampler};

#[allow(dead_code)]
pub struct Resampler {
    resampler: FftFixedIn<f32>,
    input_buffer: Vec<Vec<f32>>,
    output_buffer: Vec<Vec<f32>>,
}

#[allow(dead_code)]
impl Resampler {
    pub fn new(
        source_rate: u32,
        target_rate: u32,
        chunk_size: usize,
        channels: usize,
    ) -> Result<Self> {
        let resampler = FftFixedIn::<f32>::new(
            source_rate as usize,
            target_rate as usize,
            chunk_size,
            2, // sub_chunks
            channels,
        )?;

        let input_buffer = resampler.input_buffer_allocate(false);
        let output_buffer = resampler.output_buffer_allocate(false);

        Ok(Self {
            resampler,
            input_buffer,
            output_buffer,
        })
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // Assume mono for now or interleaved to planar
        let channels = self.input_buffer.len();
        if channels == 1 {
            self.input_buffer[0].copy_from_slice(input);
        } else {
            // De-interleave if needed
            for (i, chunk) in input.chunks(channels).enumerate() {
                for (c, &sample) in chunk.iter().enumerate() {
                    self.input_buffer[c][i] = sample;
                }
            }
        }

        self.resampler
            .process_into_buffer(&self.input_buffer, &mut self.output_buffer, None)?;

        // Interleave back if needed
        if channels == 1 {
            Ok(self.output_buffer[0].clone())
        } else {
            let n_samples = self.output_buffer[0].len();
            let mut interleaved = Vec::with_capacity(n_samples * channels);
            for i in 0..n_samples {
                for c in 0..channels {
                    interleaved.push(self.output_buffer[c][i]);
                }
            }
            Ok(interleaved)
        }
    }
}
