import time
import subprocess
from typing import Optional


class GPUCoolingCallback:
    """Add cooling intervals during training to prevent overheating"""
    
    def __init__(self, 
                 cool_every_n_epochs: int = 5,
                 cool_duration_seconds: int = 60,
                 max_temp_celsius: int = 80):
        self.cool_every = cool_every_n_epochs
        self.cool_duration = cool_duration_seconds
        self.max_temp = max_temp_celsius
        self.epoch_count = 0
        
    def get_gpu_temp(self) -> Optional[int]:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return None
    
    def should_cool(self, epoch: int) -> bool:
        # Cool every N epochs
        if (epoch + 1) % self.cool_every == 0:
            return True
        
        # Force cool if too hot
        temp = self.get_gpu_temp()
        if temp and temp >= self.max_temp:
            return True
        
        return False
    
    def cool_down(self, epoch: int, reason: str = "scheduled"):
        """Executa pausa para resfriamento"""
        temp_before = self.get_gpu_temp()
        
        print(f"\n{'='*80}")
        print(f"GPU COOLING BREAK (Epoch {epoch + 1})")
        print(f"Reason: {reason}")
        if temp_before:
            print(f"Temperature before: {temp_before}°C")
        print(f"Waiting {self.cool_duration} seconds...")
        print(f"{'='*80}\n")
        
        # Countdown visual
        for remaining in range(self.cool_duration, 0, -10):
            temp = self.get_gpu_temp()
            temp_str = f" (GPU: {temp}°C)" if temp else ""
            print(f"[WAIT] {remaining}s remaining{temp_str}", flush=True)
            time.sleep(min(10, remaining))
        
        temp_after = self.get_gpu_temp()
        
        print(f"\n{'='*80}")
        print(f"COOLING COMPLETE")
        if temp_before and temp_after:
            delta = temp_before - temp_after
            print(f"Temperature: {temp_before}°C → {temp_after}°C (Δ{delta:+d}°C)")
        print(f"Resuming training...")
        print(f"{'='*80}\n")


def inject_cooling_callback(trainer, 
                           cool_every_n_epochs: int = 5,
                           cool_duration_seconds: int = 60,
                           max_temp_celsius: int = 80):
    """
    Injeta callback de cooling no trainer do RecBole
    
    Args:
        trainer: RecBole Trainer instance
        cool_every_n_epochs: Pausar a cada N epochs
        cool_duration_seconds: Duração da pausa
        max_temp_celsius: Temperatura máxima
    """
    callback = GPUCoolingCallback(
        cool_every_n_epochs=cool_every_n_epochs,
        cool_duration_seconds=cool_duration_seconds,
        max_temp_celsius=max_temp_celsius
    )
    
    # Monkey-patch o método _train_epoch
    original_train_epoch = trainer._train_epoch
    
    def train_epoch_with_cooling(train_data, epoch_idx, loss_func=None, show_progress=False):
        # Treinar epoch normalmente
        result = original_train_epoch(train_data, epoch_idx, loss_func, show_progress)
        
        # Verificar se deve pausar
        if callback.should_cool(epoch_idx):
            temp = callback.get_gpu_temp()
            reason = "scheduled"
            if temp and temp >= callback.max_temp:
                reason = f"high temperature ({temp}°C >= {callback.max_temp}°C)"
            
            callback.cool_down(epoch_idx, reason)
        
        return result
    
    # Substituir método
    trainer._train_epoch = train_epoch_with_cooling
    
    return callback


if __name__ == '__main__':
    # Teste standalone
    callback = GPUCoolingCallback(
        cool_every_n_epochs=3,
        cool_duration_seconds=10,
        max_temp_celsius=75
    )
    
    print("Testing GPU Cooling Callback...")
    print(f"Current GPU temp: {callback.get_gpu_temp()}°C")
    
    # Simular epochs
    for epoch in range(5):
        print(f"\nEpoch {epoch}...")
        time.sleep(1)
        
        if callback.should_cool(epoch):
            callback.cool_down(epoch)
