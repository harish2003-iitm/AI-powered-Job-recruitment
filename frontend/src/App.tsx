import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import UploadJob from './pages/UploadJob';
import UploadCandidate from './pages/UploadCandidate';
import Matches from './pages/Matches';

function App() {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Navbar />
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload-job" element={<UploadJob />} />
          <Route path="/upload-candidate" element={<UploadCandidate />} />
          <Route path="/matches" element={<Matches />} />
        </Routes>
      </Box>
    </Box>
  );
}

export default App; 