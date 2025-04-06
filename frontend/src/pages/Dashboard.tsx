import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  CircularProgress,
  Alert,
  Paper,
} from '@mui/material';
import WorkIcon from '@mui/icons-material/Work';
import PersonIcon from '@mui/icons-material/Person';
import AssessmentIcon from '@mui/icons-material/Assessment';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface Stats {
  totalJobs: number;
  totalCandidates: number;
  totalMatches: number;
  highMatches: number;
}

const Dashboard = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/stats');
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error('Error fetching stats:', error);
        setError(error instanceof Error ? error.message : 'Failed to fetch statistics');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="md">
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  const chartData = stats ? [
    { name: 'Jobs', value: stats.totalJobs },
    { name: 'Candidates', value: stats.totalCandidates },
    { name: 'Matches', value: stats.totalMatches },
    { name: 'High Matches', value: stats.highMatches },
  ] : [];

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Recruitment System Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6">Total Jobs</Typography>
            <Typography variant="h4">{stats?.totalJobs || 0}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6">Total Candidates</Typography>
            <Typography variant="h4">{stats?.totalCandidates || 0}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6">Total Matches</Typography>
            <Typography variant="h4">{stats?.totalMatches || 0}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6} lg={3}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h6">High Matches</Typography>
            <Typography variant="h4">{stats?.highMatches || 0}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              System Statistics
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#1976d2" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          System Overview
        </Typography>
        <Typography variant="body1" paragraph>
          This recruitment system helps you match candidates with job positions automatically.
          Upload job descriptions and candidate CVs, and the system will:
        </Typography>
        <ul>
          <li>Parse and analyze job requirements and candidate qualifications</li>
          <li>Calculate match scores based on skills, experience, and education</li>
          <li>Generate interview requests for high-matching candidates</li>
          <li>Track and manage the entire recruitment process</li>
        </ul>
      </Box>
    </Container>
  );
};

export default Dashboard; 