import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np

class GrapeGAT(torch.nn.Module):
    """
    Graph Attention Network for retinal vessel classification.
    - GAT layers with multi-head attention
    - Multiple pooling strategies (mean + max + add)
    - Graph-level topological features
    """
    def __init__(self, in_dim, hid=64, out=2, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # GAT layers with multi-head attention
        self.conv1 = GATConv(in_dim, hid, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hid * heads, hid, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hid * heads, hid, heads=1, dropout=dropout)
        
        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(hid * heads)
        self.bn2 = torch.nn.BatchNorm1d(hid * heads)
        self.bn3 = torch.nn.BatchNorm1d(hid)
        
        # MLP classifier (3 pooling strategies * hid + graph features)
        self.fc1 = torch.nn.Linear(hid * 3 + 5, hid)
        self.fc2 = torch.nn.Linear(hid, out)
    
    def forward(self, x, edge_index, batch, graph_feats):
        # GAT convolutions with residual-like structure
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        
        # Multiple pooling strategies
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        # Concatenate pooled features with graph-level features
        x = torch.cat([x_mean, x_max, x_add, graph_feats], dim=1)
        
        # MLP classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)

def compute_graph_features(g, node_map, edges):
    """Compute graph-level topological features"""
    num_nodes = len(g)
    num_edges = len(edges)
    
    # Compute degree statistics
    degree = np.zeros(num_nodes)
    for e in edges:
        degree[e[0]] += 1
    
    avg_degree = degree.mean() if num_nodes > 0 else 0
    max_degree = degree.max() if num_nodes > 0 else 0
    
    # Node type distribution
    junction_ratio = (g['type'] == 'junction').mean() if 'type' in g.columns else 0
    
    # Average vessel width
    avg_width = g['width'].mean() if 'width' in g.columns else 0
    
    return [num_nodes / 500, num_edges / 500, avg_degree / 5, junction_ratio, avg_width / 10]

def load_graphs(graph_path, label_path=None):
    df = pd.read_csv(graph_path)
    labels = pd.read_csv(label_path) if label_path else None
    graphs = []
    
    for gid in df['graph_id'].unique():
        g = df[df['graph_id']==gid].reset_index(drop=True)
        
        # Node features: x, y, width + node type encoding
        node_type_map = {'junction': 0, 'endpoint': 1}
        type_vals = g['type'].map(lambda t: node_type_map.get(t, 0)).values if 'type' in g.columns else np.zeros(len(g))
        
        x = np.column_stack([
            g['x'].values / 600,
            g['y'].values / 600,
            g['width'].values / 20,
            type_vals,  # Node type
        ])
        x = torch.tensor(x, dtype=torch.float)
        
        # Build edge list
        edges = []
        node_map = {row['node_id']: i for i, row in g.iterrows()}
        for i, row in g.iterrows():
            if pd.notna(row['edges']) and row['edges']:
                for tgt in str(row['edges']).split(';'):
                    if tgt.strip().isdigit():
                        tgt_id = int(tgt)
                        if tgt_id in node_map:
                            edges.append([i, node_map[tgt_id]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2,0,dtype=torch.long)
        
        # Compute graph-level features
        graph_feats = compute_graph_features(g, node_map, edges)
        
        y = torch.tensor([labels[labels['graph_id']==gid]['label'].values[0]]) if labels is not None else None
        data = Data(x=x, edge_index=edge_index, y=y)
        data.gid = gid
        data.graph_feats = torch.tensor([graph_feats], dtype=torch.float)
        graphs.append(data)
    
    return graphs

def train():
    graphs = load_graphs('data/public/train_data.csv', 'data/public/train_labels.csv')
    
    # Split into train/val (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    val_size = int(0.2 * len(graphs))
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Compute class weights for imbalanced data
    labels = [g.y.item() for g in train_graphs]
    class_counts = np.bincount(labels)
    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    print(f"Class weights: {class_weights.tolist()}")
    
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    model = GrapeGAT(in_dim=4, hid=64, out=2, heads=4, dropout=0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    
    best_val_acc = 0
    patience = 30
    patience_counter = 0
    best_model_state = None
    
    for ep in range(300):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            opt.zero_grad()
            graph_feats = torch.cat([train_graphs[i].graph_feats for i in range(len(batch.y))], dim=0)
            out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
            loss = F.cross_entropy(out, batch.y, weight=class_weights)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += len(batch.y)
        
        scheduler.step()
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                graph_feats = torch.cat([val_graphs[i].graph_feats for i in range(len(batch.y))], dim=0)
                out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += len(batch.y)
        
        val_acc = val_correct / val_total
        
        # Early stopping on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1}: loss={total_loss/len(train_loader):.4f}, train_acc={train_acc*100:.1f}%, val_acc={val_acc*100:.1f}%")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    return model

def predict(model, graph_path, out_path):
    graphs = load_graphs(graph_path)
    preds = []
    model.eval()
    
    with torch.no_grad():
        for g in graphs:
            out = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long), g.graph_feats)
            pred = out.argmax(dim=1).item()
            preds.append({'graph_id': g.gid, 'label': pred})
    
    pd.DataFrame(preds).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    
    # Show prediction distribution
    pred_df = pd.DataFrame(preds)
    print(f"Prediction distribution: {pred_df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    model = train()
    predict(model, 'data/public/test_data.csv', 'submission.csv')
