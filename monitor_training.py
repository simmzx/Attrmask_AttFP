#!/usr/bin/env python
"""
Real-time Training Monitor
==========================
Monitor training progress in real-time from log files.

Usage:
    # Terminal 1: Run training
    python pretrain_attrmasking_enhanced.py --dataset data.txt --use_tensorboard
    
    # Terminal 2: Monitor
    python monitor_training.py --log_file checkpoints/logs/attrmasking_pretrain.log
    
    # Or use TensorBoard (recommended)
    tensorboard --logdir=checkpoints/tensorboard
"""

import argparse
import os
import time
import re
import sys

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def parse_log_line(line):
    """Parse a log line to extract metrics"""
    # Pattern: Epoch X Step Y | Loss: Z.ZZZZ | Acc: Z.Z%
    pattern = r'Epoch (\d+).*Loss: ([\d.]+).*Acc: ([\d.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'loss': float(match.group(2)),
            'acc': float(match.group(3))
        }
    return None

def tail_file(filename, n=10):
    """Get last n lines of a file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def monitor_log(log_file, update_interval=5, show_plot=False):
    """Monitor log file in real-time"""
    print(f"Monitoring: {log_file}")
    print(f"Update interval: {update_interval}s")
    print("Press Ctrl+C to stop\n")
    print("-" * 70)
    
    losses = []
    accs = []
    epochs = []
    
    last_size = 0
    
    if show_plot and HAS_MATPLOTLIB:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    try:
        while True:
            if not os.path.exists(log_file):
                print(f"Waiting for log file: {log_file}")
                time.sleep(update_interval)
                continue
            
            current_size = os.path.getsize(log_file)
            
            if current_size != last_size:
                last_size = current_size
                
                # Read new lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse metrics
                for line in lines:
                    metrics = parse_log_line(line)
                    if metrics:
                        if len(epochs) == 0 or metrics['epoch'] > epochs[-1]:
                            epochs.append(metrics['epoch'])
                            losses.append(metrics['loss'])
                            accs.append(metrics['acc'])
                
                # Clear screen and show latest
                os.system('clear' if os.name == 'posix' else 'cls')
                print("=" * 70)
                print("AttrMasking Training Monitor")
                print("=" * 70)
                print(f"Log file: {log_file}")
                print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 70)
                
                if len(epochs) > 0:
                    print(f"\nLatest Metrics:")
                    print(f"  Epoch: {epochs[-1]}")
                    print(f"  Loss:  {losses[-1]:.4f}")
                    print(f"  Acc:   {accs[-1]:.1f}%")
                    
                    if len(losses) > 1:
                        print(f"\nProgress:")
                        print(f"  Best Loss:  {min(losses):.4f} (Epoch {epochs[losses.index(min(losses))]})")
                        print(f"  Best Acc:   {max(accs):.1f}% (Epoch {epochs[accs.index(max(accs))]})")
                        
                        # Show trend
                        if len(losses) >= 5:
                            recent_loss = sum(losses[-5:]) / 5
                            older_loss = sum(losses[-10:-5]) / 5 if len(losses) >= 10 else losses[0]
                            if recent_loss < older_loss:
                                print(f"  Trend:      ↓ Loss decreasing (good!)")
                            else:
                                print(f"  Trend:      ↑ Loss increasing")
                
                print("\n" + "-" * 70)
                print("Recent log entries:")
                for line in tail_file(log_file, 5):
                    print(f"  {line.strip()}")
                
                # Update plot
                if show_plot and HAS_MATPLOTLIB and len(epochs) > 1:
                    ax1.clear()
                    ax2.clear()
                    
                    ax1.plot(epochs, losses, 'b-o', markersize=3)
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Training Loss')
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(epochs, accs, 'g-o', markersize=3)
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy (%)')
                    ax2.set_title('Training Accuracy')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.pause(0.1)
            
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if show_plot and HAS_MATPLOTLIB:
            plt.ioff()
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Training Monitor')
    parser.add_argument('--log_file', type=str, 
                        default='checkpoints/logs/attrmasking_pretrain.log',
                        help='Path to log file')
    parser.add_argument('--interval', type=int, default=5,
                        help='Update interval in seconds')
    parser.add_argument('--plot', action='store_true',
                        help='Show real-time plot (requires matplotlib)')
    args = parser.parse_args()
    
    monitor_log(args.log_file, args.interval, args.plot)

if __name__ == "__main__":
    main()
