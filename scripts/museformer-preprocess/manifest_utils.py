"""
Manifest Management Utilities for MuseFormer Preprocessing Pipeline

This module provides utilities for managing the manifest CSV that tracks
files through all preprocessing stages.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ManifestManager:
    """Manages the preprocessing manifest CSV file."""
    
    def __init__(self, manifest_path: Path):
        """
        Initialize manifest manager.
        
        Args:
            manifest_path: Path to manifest CSV file
        """
        self.manifest_path = Path(manifest_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """
        Load manifest from CSV.
        
        Returns:
            DataFrame containing manifest data
        """
        if not self.manifest_path.exists():
            logger.warning(f"Manifest not found: {self.manifest_path}")
            # Create empty manifest with basic columns
            self.df = pd.DataFrame(columns=[
                'file_id', 'raw_path', 'raw_basename', 'stage', 'status', 
                'drop_reason', 'error_msg'
            ])
        else:
            self.df = pd.read_csv(self.manifest_path)
            logger.info(f"Loaded manifest: {len(self.df)} rows from {self.manifest_path}")
        
        # Ensure raw_basename column exists
        if 'raw_basename' not in self.df.columns and 'raw_path' in self.df.columns:
            self.df['raw_basename'] = self.df['raw_path'].astype(str).map(
                lambda p: Path(p).name if pd.notna(p) else ''
            )
        
        return self.df
    
    def save(self, backup: bool = True) -> None:
        """
        Save manifest to CSV.
        
        Args:
            backup: Whether to create a backup before saving
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        # Create backup if requested and file exists
        if backup and self.manifest_path.exists():
            backup_path = self.manifest_path.with_suffix(
                f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.manifest_path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Ensure parent directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        self.df.to_csv(self.manifest_path, index=False)
        logger.info(f"Saved manifest: {len(self.df)} rows to {self.manifest_path}")
    
    def update_rows(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """
        Update manifest rows for specific files.
        
        Args:
            updates: Dict mapping basename -> dict of column values
                    Example: {'file1.mid': {'stage': '01_parsing', 'status': 'ok'}}
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        # Add any new columns that don't exist
        all_cols = set(k for u in updates.values() for k in u.keys())
        for col in all_cols:
            if col not in self.df.columns:
                self.df[col] = pd.NA
                logger.debug(f"Added new column: {col}")
        
        # Apply updates
        for basename, cols in updates.items():
            mask = self.df['raw_basename'].astype(str) == basename
            if not mask.any():
                logger.warning(f"File not found in manifest: {basename}")
                continue
            
            for col, value in cols.items():
                self.df.loc[mask, col] = value
        
        logger.info(f"Updated {len(updates)} rows in manifest")
    
    def update_rows_by_file_id(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """
        Update manifest rows for specific files using file_id as key.
        
        Args:
            updates: Dict mapping file_id -> dict of column values
                    Example: {'0/file1': {'stage': '01_parsing', 'status': 'ok'}}
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        # Add any new columns that don't exist
        all_cols = set(k for u in updates.values() for k in u.keys())
        for col in all_cols:
            if col not in self.df.columns:
                self.df[col] = pd.NA
                logger.debug(f"Added new column: {col}")
        
        # Apply updates using file_id
        for file_id, cols in updates.items():
            mask = self.df['file_id'].astype(str) == file_id
            if not mask.any():
                logger.warning(f"File ID not found in manifest: {file_id}")
                continue
            
            for col, value in cols.items():
                self.df.loc[mask, col] = value
        
        logger.info(f"Updated {len(updates)} rows in manifest")
    
    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        """
        Add new rows to manifest.
        
        Args:
            rows: List of dicts, each representing a row to add
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        new_df = pd.DataFrame(rows)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        logger.info(f"Added {len(rows)} rows to manifest")
    
    def query_by_stage(self, stage: str, status: Optional[str] = None) -> pd.DataFrame:
        """
        Query files at a specific stage.
        
        Args:
            stage: Stage name (e.g., '01_parsed_meta', '02_b_midiminer')
            status: Optional status filter ('ok', 'fail', 'drop')
        
        Returns:
            DataFrame of matching rows
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        mask = self.df['stage'].astype(str) == stage
        
        if status is not None:
            mask &= self.df['status'].astype(str) == status
        
        result = self.df[mask]
        logger.debug(f"Query stage={stage}, status={status}: {len(result)} rows")
        return result
    
    def query_by_status(self, status: str) -> pd.DataFrame:
        """
        Query files by status.
        
        Args:
            status: Status to filter by ('ok', 'fail', 'drop')
        
        Returns:
            DataFrame of matching rows
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        mask = self.df['status'].astype(str) == status
        result = self.df[mask]
        logger.debug(f"Query status={status}: {len(result)} rows")
        return result
    
    def get_file_info(self, basename: str) -> Optional[pd.Series]:
        """
        Get information for a specific file.
        
        Args:
            basename: File basename (e.g., 'file123.mid')
        
        Returns:
            Series with file info, or None if not found
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        mask = self.df['raw_basename'].astype(str) == basename
        matches = self.df[mask]
        
        if len(matches) == 0:
            return None
        elif len(matches) > 1:
            logger.warning(f"Multiple entries found for {basename}, returning first")
        
        return matches.iloc[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from manifest.
        
        Returns:
            Dict with statistics
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        stats = {
            'total_files': len(self.df),
            'by_status': self.df['status'].value_counts().to_dict(),
            'by_stage': self.df['stage'].value_counts().to_dict(),
        }
        
        # Count drop reasons
        if 'drop_reason' in self.df.columns:
            drop_mask = self.df['status'].astype(str) == 'drop'
            if drop_mask.any():
                stats['drop_reasons'] = self.df[drop_mask]['drop_reason'].value_counts().to_dict()
        
        return stats
    
    def validate(self) -> List[str]:
        """
        Validate manifest integrity.
        
        Returns:
            List of validation errors (empty if valid)
        """
        if self.df is None:
            raise ValueError("No manifest loaded. Call load() first.")
        
        errors = []
        
        # Check required columns
        required_cols = ['file_id', 'raw_path', 'raw_basename', 'stage', 'status']
        for col in required_cols:
            if col not in self.df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for duplicate file_ids
        if 'file_id' in self.df.columns:
            duplicates = self.df['file_id'].duplicated()
            if duplicates.any():
                dup_ids = self.df[duplicates]['file_id'].tolist()
                errors.append(f"Duplicate file_ids found: {dup_ids[:5]}")
        
        # Check for missing basenames
        if 'raw_basename' in self.df.columns:
            missing = self.df['raw_basename'].isna() | (self.df['raw_basename'] == '')
            if missing.any():
                errors.append(f"Missing basenames: {missing.sum()} rows")
        
        return errors


def create_manifest_from_files(
    file_paths: List[Path],
    manifest_path: Path,
    file_id_func=None
) -> ManifestManager:
    """
    Create a new manifest from a list of files.
    
    Args:
        file_paths: List of file paths to include
        manifest_path: Path where manifest will be saved
        file_id_func: Optional function to generate file_id from path
                     Default: uses MD5 hash of relative path
    
    Returns:
        ManifestManager with populated manifest
    """
    import hashlib
    
    if file_id_func is None:
        def file_id_func(p):
            rel = p.name
            return hashlib.md5(rel.encode('utf-8')).hexdigest()[:12]
    
    rows = []
    for path in file_paths:
        rows.append({
            'file_id': file_id_func(path),
            'raw_path': str(path),
            'raw_basename': path.name,
            'stage': 'initial',
            'status': 'pending',
            'drop_reason': '',
            'error_msg': '',
        })
    
    manager = ManifestManager(manifest_path)
    manager.df = pd.DataFrame(rows)
    manager.save(backup=False)
    
    logger.info(f"Created manifest with {len(rows)} files")
    return manager
