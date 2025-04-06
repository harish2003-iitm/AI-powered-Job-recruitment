import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const UploadCandidate = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    const validFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    
    if (validFiles.length === selectedFiles.length) {
      setFiles(validFiles);
      setError(null);
    } else {
      setError('Please select valid PDF files');
      setFiles([]);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select files to upload');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(false);

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://127.0.0.1:8000/upload/candidates', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload candidate CVs');
      }

      setSuccess(true);
      setFiles([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload candidate CVs');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Upload Candidate CVs
      </Typography>
      <Paper sx={{ p: 3 }}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <input
            accept=".pdf"
            style={{ display: 'none' }}
            id="candidate-file-upload"
            type="file"
            multiple
            onChange={handleFileChange}
          />
          <label htmlFor="candidate-file-upload">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUploadIcon />}
            >
              Select PDF Files
            </Button>
          </label>
          {files.length > 0 && (
            <Typography variant="body1">
              Selected {files.length} file(s)
            </Typography>
          )}
          {error && (
            <Alert severity="error" sx={{ width: '100%' }}>
              {error}
            </Alert>
          )}
          {success && (
            <Alert severity="success" sx={{ width: '100%' }}>
              Candidate CVs uploaded successfully!
            </Alert>
          )}
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
            startIcon={uploading ? <CircularProgress size={20} /> : null}
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default UploadCandidate; 