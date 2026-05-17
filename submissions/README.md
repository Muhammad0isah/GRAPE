Place encrypted submissions in `inbox/<team>/submission.csv.enc`

Encrypt with: `python encryption/encrypt.py submission.csv submissions/inbox/<team>/submission.csv.enc`

### meta.yaml

Create a `meta.yaml` file in your team folder alongside the `.enc` file:

```yaml
model: VesselGCN                          # Name of your model (optional)
type: human                               # human, llm, or human+llm
notes: 3-layer GCN with skip connections  # Brief description (optional)
```

Example folder structure:

```
submissions/inbox/<team>/
├── submission.csv.enc   # Required (encrypted predictions)
└── meta.yaml            # Describes your submission
```
