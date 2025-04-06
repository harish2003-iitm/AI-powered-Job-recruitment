import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';

interface Match {
  id: number;
  candidate_id: number;
  candidate_name: string;
  job_id: number;
  job_title: string;
  overall_score: number;
  skills_match: number;
  experience_match: number;
  education_match: number;
  candidate_email?: string;
  created_at?: string;
}

interface InterviewRequest {
  subject: string;
  body: string;
  proposed_dates: string[];
  interview_type: string;
}

const Matches = () => {
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<Match | null>(null);
  const [interviewRequest, setInterviewRequest] = useState<InterviewRequest | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  useEffect(() => {
    fetchMatches();
  }, []);

  const fetchMatches = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/matches');
      if (!response.ok) {
        throw new Error('Failed to fetch matches');
      }
      const data = await response.json();
      setMatches(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch matches');
    } finally {
      setLoading(false);
    }
  };

  const handleViewInterview = async (match: Match) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/candidates/${match.candidate_id}/interview-request?job_id=${match.job_id}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          setError(`No interview request found for this candidate and job combination.`);
          return;
        }
        throw new Error('Failed to fetch interview request');
      }
      
      const data = await response.json();
      setInterviewRequest(data);
      setSelectedMatch(match);
      setDialogOpen(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch interview request');
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Matches
      </Typography>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Candidate</TableCell>
              <TableCell>Job Title</TableCell>
              <TableCell align="right">Overall Score</TableCell>
              <TableCell align="right">Skills Match</TableCell>
              <TableCell align="right">Experience Match</TableCell>
              <TableCell align="right">Education Match</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {matches.map((match) => (
              <TableRow key={match.id}>
                <TableCell>{match.candidate_name}</TableCell>
                <TableCell>{match.job_title}</TableCell>
                <TableCell align="right">{match.overall_score !== undefined ? match.overall_score.toFixed(0) : 'N/A'}%</TableCell>
                <TableCell align="right">{match.skills_match !== undefined ? match.skills_match.toFixed(0) : 'N/A'}%</TableCell>
                <TableCell align="right">{match.experience_match !== undefined ? match.experience_match.toFixed(0) : 'N/A'}%</TableCell>
                <TableCell align="right">{match.education_match !== undefined ? match.education_match.toFixed(0) : 'N/A'}%</TableCell>
                <TableCell align="center">
                  <Button
                    startIcon={<VisibilityIcon />}
                    onClick={() => handleViewInterview(match)}
                    disabled={match.overall_score < 80}
                  >
                    View Interview
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Interview Request for {selectedMatch?.candidate_name}
        </DialogTitle>
        <DialogContent>
          {interviewRequest && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Subject
              </Typography>
              <Typography paragraph>{interviewRequest.subject}</Typography>
              
              <Typography variant="h6" gutterBottom>
                Body
              </Typography>
              <Typography paragraph>{interviewRequest.body}</Typography>
              
              <Typography variant="h6" gutterBottom>
                Proposed Dates
              </Typography>
              <Typography paragraph>
                {interviewRequest.proposed_dates.join(', ')}
              </Typography>
              
              <Typography variant="h6" gutterBottom>
                Interview Type
              </Typography>
              <Typography paragraph>{interviewRequest.interview_type}</Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Matches; 