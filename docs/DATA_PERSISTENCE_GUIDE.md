# Data Persistence Guide

## Overview

The HFT Arbitrage Lab now supports **persistent data storage** using Docker volumes. All datasets loaded through the Data Loader are automatically saved to disk and persist across application restarts and Docker container recreations.

## Features

### 🔄 Automatic Persistence
- **Auto-save**: Every dataset fetched is automatically saved to persistent storage
- **Append Mode**: New data can be appended to existing datasets
- **Deduplication**: Automatically removes duplicate entries based on symbol and timestamp
- **Efficient Storage**: Uses Parquet format for optimal compression and performance

### 💾 Storage Location

**Local Development:**
```
data/persisted/
├── metadata.json           # Dataset metadata and index
├── dataset_name1.parquet   # Saved dataset files
├── dataset_name2.parquet
└── ...
```

**Docker Deployment:**
- Data is stored in named Docker volumes
- Persists even when containers are stopped or removed
- Volumes: `hft_data`, `hft_data_standalone`, or `hft_data_prod`

## Using Persistent Data

### 1. Loading Data

Navigate to **📊 Data & Market → Data Loader**:

1. **View Saved Datasets**: Click the "💾 Persisted Datasets" expander at the top
2. **Load Dataset**: Click "📂 Load" button next to any saved dataset
3. **Auto-load**: The homepage shows available datasets on startup

### 2. Saving Data

After fetching data:

1. Go to the **💾 Export** tab in the data preview
2. Enter a **Dataset Name**
3. Choose **Save Mode**:
   - **Create New**: Creates a fresh dataset
   - **Append to Existing**: Adds to existing dataset with same name
4. Click **💾 Save Dataset**

### 3. Managing Datasets

In the **💾 Persisted Datasets** section:

- **View Statistics**: See total datasets, rows, symbols, and storage size
- **Load**: Load any dataset into current session
- **Delete**: Remove unwanted datasets
- **Refresh**: Update the dataset list

## Docker Volume Management

### View Volume Data

```bash
# List Docker volumes
docker volume ls

# Inspect volume details
docker volume inspect hft_data

# View volume path (on Docker host)
docker volume inspect hft_data -f '{{.Mountpoint}}'
```

### Backup Volume Data

```bash
# Backup to tar archive
docker run --rm -v hft_data:/data -v $(pwd):/backup alpine tar czf /backup/hft_data_backup.tar.gz -C /data .

# Restore from backup
docker run --rm -v hft_data:/data -v $(pwd):/backup alpine tar xzf /backup/hft_data_backup.tar.gz -C /data
```

### Access Volume Data

```bash
# Copy data out of volume
docker run --rm -v hft_data:/data -v $(pwd)/backup:/backup alpine cp -r /data /backup

# Copy data into volume
docker run --rm -v hft_data:/data -v $(pwd)/backup:/backup alpine cp -r /backup/data/* /data/
```

### Clean Up Volumes

```bash
# Remove specific volume (WARNING: Deletes all data!)
docker volume rm hft_data

# Remove all unused volumes
docker volume prune
```

## API Reference

### Python API

```python
from utils.data_persistence import (
    save_dataset,
    load_dataset,
    load_all_datasets,
    delete_dataset,
    list_datasets,
    get_storage_stats,
    merge_datasets
)

# Save a dataset
save_dataset(
    df=dataframe,
    dataset_name="my_stocks",
    symbols=["AAPL", "GOOGL"],
    source="Yahoo Finance",
    date_range=("2024-01-01", "2024-12-01"),
    append=False  # or True to append
)

# Load a dataset
df = load_dataset("my_stocks")

# Load all datasets
all_datasets = load_all_datasets()

# Get dataset info
datasets = list_datasets()
stats = get_storage_stats()

# Delete a dataset
delete_dataset("my_stocks")

# Merge multiple datasets
merge_datasets(
    dataset_names=["dataset1", "dataset2"],
    new_dataset_name="combined_dataset"
)
```

## Best Practices

### 1. **Naming Conventions**
- Use descriptive names: `spy_2024_daily`, `btc_1h_2024`, etc.
- Include timeframe and data type in name
- Avoid special characters (use underscore or hyphen)

### 2. **Storage Management**
- Regularly review and delete unused datasets
- Use append mode to incrementally update datasets
- Monitor storage size on the homepage

### 3. **Data Organization**
- Group related symbols in named datasets
- Keep different timeframes separate
- Tag datasets with source information

### 4. **Backup Strategy**
- Regularly backup Docker volumes (see commands above)
- Export critical datasets as CSV/Parquet
- Use version control for dataset metadata

## Configuration

### Docker Compose Files

All three deployment modes include persistent storage:

**docker-compose.yml** (Development)
```yaml
volumes:
  - hft_data:/app/data
volumes:
  hft_data:
    driver: local
```

**docker-compose.standalone.yml**
```yaml
volumes:
  - hft_data_standalone:/app/data
volumes:
  hft_data_standalone:
    driver: local
```

**docker-compose.prod.yml** (Production)
```yaml
volumes:
  - hft_data_prod:/app/data
volumes:
  hft_data_prod:
    driver: local
```

## Troubleshooting

### Dataset Not Showing Up

1. Check if data directory exists: `ls -la data/persisted/`
2. Verify Docker volume is mounted: `docker inspect <container_name>`
3. Refresh dataset list from homepage or data loader

### Storage Full

1. Check storage stats on homepage
2. Delete unused datasets
3. Export and archive old datasets
4. Increase Docker volume size if needed

### Permission Errors

```bash
# Fix permissions (if running locally)
chmod -R 755 data/persisted/

# In Docker, ensure volume has correct permissions
docker run --rm -v hft_data:/data alpine chmod -R 755 /data
```

### Data Lost After Container Restart

- Verify named volumes are used (not bind mounts)
- Check `docker volume ls` to confirm volume exists
- Ensure volume is properly mapped in docker-compose.yml

## Migration

### From Session State to Persistent Storage

If you have data in session state:

1. Go to Data Loader
2. View your loaded data
3. Navigate to **💾 Export** tab
4. Enter a name and click **Save Dataset**

### Between Docker Deployments

```bash
# Export from development
docker run --rm -v hft_data:/data -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz -C /data .

# Import to production
docker run --rm -v hft_data_prod:/data -v $(pwd):/backup alpine tar xzf /backup/data_backup.tar.gz -C /data
```

## Performance Tips

1. **Parquet Format**: Already optimized for fast I/O
2. **Compression**: Snappy compression is used by default
3. **Chunking**: Large datasets are efficiently handled
4. **Deduplication**: Runs automatically to minimize storage

## Future Enhancements

- [ ] Cloud storage integration (S3, GCS)
- [ ] Dataset versioning
- [ ] Automatic compression of old data
- [ ] Dataset sharing between users
- [ ] Real-time sync across instances

## Support

For issues or questions:
- Check existing datasets: Visit Data Loader → Persisted Datasets
- Review logs: `docker logs <container_name>`
- Check documentation: `docs/` directory
- GitHub Issues: Report bugs or request features

---

**Note**: Persistent storage is designed for convenience during development and research. For production deployments, consider implementing proper database solutions and backup strategies.
