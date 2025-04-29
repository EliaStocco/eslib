from tqdm import tqdm
import time

# Create 4 progress bars with different positions
bars = [tqdm(total=100, position=i, desc=f'Bar {i+1}') for i in range(4)]

for i in range(100):
    for bar in bars:
        bar.update(1)
        time.sleep(0.01)  # Just for visual effect

