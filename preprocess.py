import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.morphology import skeletonize
from scipy import ndimage

TARGET_SIZE = 600  # Resize large images to this max dimension

def load_mask(path, resize=True):
    """Load vessel segmentation mask from various formats."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load: {path}")
    # Resize if too large
    if resize and max(img.shape) > TARGET_SIZE:
        scale = TARGET_SIZE / max(img.shape)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return (img > 127).astype(np.uint8)

def get_neighbors(skel, y, x):
    h, w = skel.shape
    count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                count += 1
    return count

def find_keypoints(skel):
    h, w = skel.shape
    endpoints, junctions = [], []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if not skel[y, x]: continue
            n = get_neighbors(skel, y, x)
            if n == 1: endpoints.append((y, x))
            elif n >= 3: junctions.append((y, x))
    return endpoints, junctions

def merge_close(points, thresh=10):
    if not points: return []
    merged, used = [], set()
    for i, p1 in enumerate(points):
        if i in used: continue
        cluster = [p1]
        used.add(i)
        for j, p2 in enumerate(points):
            if j in used: continue
            if np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < thresh:
                cluster.append(p2)
                used.add(j)
        merged.append((np.mean([p[0] for p in cluster]), np.mean([p[1] for p in cluster])))
    return merged

def trace_edge(skel, start, end, max_steps=500):
    from collections import deque
    h, w = skel.shape
    visited = set([start])
    queue = deque([(start, [start])])
    while queue and len(visited) < max_steps:
        (y, x), path = queue.popleft()
        if abs(y - end[0]) <= 5 and abs(x - end[1]) <= 5:
            return path
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append(((ny, nx), path + [(ny, nx)]))
    return None

def estimate_width(mask, skel, y, x):
    dist = ndimage.distance_transform_edt(mask)
    if 0 <= int(y) < dist.shape[0] and 0 <= int(x) < dist.shape[1]:
        return float(dist[int(y), int(x)] * 2)
    return 3.0

def mask_to_graph(mask_path, graph_id):
    mask = load_mask(mask_path)
    skel = skeletonize(mask > 0).astype(np.uint8)
    
    endpoints, junctions = find_keypoints(skel)
    endpoints = merge_close(endpoints, 15)
    junctions = merge_close(junctions, 15)
    
    nodes = []
    for i, (y, x) in enumerate(junctions):
        nodes.append({'id': i, 'y': y, 'x': x, 'type': 1, 'width': estimate_width(mask, skel, y, x)})
    offset = len(junctions)
    for i, (y, x) in enumerate(endpoints):
        nodes.append({'id': offset+i, 'y': y, 'x': x, 'type': 0, 'width': estimate_width(mask, skel, y, x)})
    
    edges = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j: continue
            dist = np.sqrt((n1['y']-n2['y'])**2 + (n1['x']-n2['x'])**2)
            if dist < 80:
                path = trace_edge(skel, (int(n1['y']), int(n1['x'])), (int(n2['y']), int(n2['x'])))
                if path and len(path) > 5:
                    edges.append((i, j, len(path)))
    
    adj = {i: [] for i in range(len(nodes))}
    for src, tgt, _ in edges:
        adj[src].append(tgt)
        adj[tgt].append(src)
    
    rows = []
    for n in nodes:
        rows.append({
            'graph_id': graph_id,
            'node_id': n['id'],
            'x': round(n['x'], 1),
            'y': round(n['y'], 1),
            'width': round(n['width'], 1),
            'type': n['type'],
            'edges': ';'.join(map(str, adj[n['id']])) if adj[n['id']] else ''
        })
    return rows

def load_stare_diagnoses(path):
    """Load STARE diagnosis codes. Code 7 = Diabetic Retinopathy."""
    diagnoses = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_id = parts[0].strip()
                codes = parts[1].strip().split()
                has_dr = '7' in codes
                diagnoses[img_id] = 1 if has_dr else 0
    return diagnoses

def main():
    data_dir = Path(__file__).parent / 'data'
    out_dir = data_dir / 'public'
    
    all_rows = []
    all_labels = []
    
    # === DRIVE dataset (images 21-40) ===
    print("Processing DRIVE dataset...")
    drive_masks = sorted((data_dir / 'raw' / 'drive').glob('*_manual1.gif'))
    
    drive_dr = {21:0, 22:0, 23:0, 24:0, 25:1, 26:1, 27:0, 28:0, 29:0, 30:0,
                31:0, 32:1, 33:0, 34:0, 35:0, 36:0, 37:0, 38:0, 39:0, 40:0}
    
    for mask_path in drive_masks:
        img_num = int(mask_path.stem.split('_')[0])
        gid = f"D_{img_num}"
        print(f"  {gid}...")
        rows = mask_to_graph(str(mask_path), gid)
        all_rows.extend(rows)
        all_labels.append({'graph_id': gid, 'label': drive_dr.get(img_num, 0)})
    
    # === STARE dataset ===
    print("\nProcessing STARE dataset...")
    stare_dir = data_dir / 'raw' / 'stare'
    stare_diagnoses = load_stare_diagnoses(data_dir / 'hrf' / 'stare_codes.txt') if (data_dir / 'hrf' / 'stare_codes.txt').exists() else {}
    
    stare_ids = [1, 2, 3, 4, 5, 44, 77, 81, 82, 139, 162, 163, 235, 236, 239, 240, 255, 291, 319, 324]
    # STARE DR labels (code 7 in original)
    stare_dr = {1:1, 9:1, 13:1, 16:1}  # Images with Background DR
    
    for img_num in stare_ids:
        mask_path = stare_dir / f"im{img_num:04d}.ah.ppm"
        if not mask_path.exists():
            print(f"  Skipping S_{img_num} (no mask)")
            continue
        gid = f"S_{img_num}"
        print(f"  {gid}...")
        rows = mask_to_graph(str(mask_path), gid)
        all_rows.extend(rows)
        label = stare_dr.get(img_num, 0)
        all_labels.append({'graph_id': gid, 'label': label})
    
    # === HRF dataset ===
    print("\nProcessing HRF dataset...")
    hrf_dir = data_dir / 'raw' / 'hrf'
    
    # Healthy images (01-15)
    for i in range(1, 16):
        mask_path = hrf_dir / f"{i:02d}_h.tif"
        if not mask_path.exists():
            continue
        gid = f"H_{i}"
        print(f"  {gid}...")
        rows = mask_to_graph(str(mask_path), gid)
        all_rows.extend(rows)
        all_labels.append({'graph_id': gid, 'label': 0})
    
    # DR images (01-15)
    for i in range(1, 16):
        mask_path = hrf_dir / f"{i:02d}_dr.tif"
        if not mask_path.exists():
            continue
        gid = f"R_{i}"  # R for Retinopathy
        print(f"  {gid}...")
        rows = mask_to_graph(str(mask_path), gid)
        all_rows.extend(rows)
        all_labels.append({'graph_id': gid, 'label': 1})
    
    # === Split into train/test ===
    print(f"\nTotal: {len(all_labels)} graphs")
    
    # Stratified split - ensure balanced test set
    dr_indices = [i for i, l in enumerate(all_labels) if l['label'] == 1]
    healthy_indices = [i for i, l in enumerate(all_labels) if l['label'] == 0]
    
    np.random.seed(42)
    np.random.shuffle(dr_indices)
    np.random.shuffle(healthy_indices)
    
    # Test: 5 DR + 10 healthy = 15
    # Train: rest
    test_idx = dr_indices[:5] + healthy_indices[:10]
    train_idx = dr_indices[5:] + healthy_indices[10:]
    
    train_labels = [all_labels[i] for i in train_idx]
    test_labels = [all_labels[i] for i in test_idx]
    train_gids = set(l['graph_id'] for l in train_labels)
    test_gids = set(l['graph_id'] for l in test_labels)
    
    train_rows = [r for r in all_rows if r['graph_id'] in train_gids]
    test_rows = [r for r in all_rows if r['graph_id'] in test_gids]
    
    # Save files
    pd.DataFrame(train_rows).to_csv(out_dir / 'train_data.csv', index=False)
    pd.DataFrame(train_labels).to_csv(out_dir / 'train_labels.csv', index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / 'test_data.csv', index=False)
    pd.DataFrame(test_labels).to_csv(data_dir / 'private' / 'test_labels.csv', index=False)
    pd.DataFrame([{'graph_id': l['graph_id'], 'label': 0} for l in test_labels]).to_csv(out_dir / 'sample_submission.csv', index=False)
    
    print(f"\n=== Summary ===")
    print(f"Train: {len(train_labels)} graphs, {len(train_rows)} nodes")
    print(f"Test:  {len(test_labels)} graphs, {len(test_rows)} nodes")
    print(f"Train DR: {sum(l['label'] for l in train_labels)}/{len(train_labels)}")
    print(f"Test DR:  {sum(l['label'] for l in test_labels)}/{len(test_labels)}")

if __name__ == "__main__":
    main()
