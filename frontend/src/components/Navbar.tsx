import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
} from '@mui/material';

const Navbar = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography
          variant="h6"
          component={RouterLink}
          to="/"
          sx={{
            flexGrow: 1,
            textDecoration: 'none',
            color: 'inherit',
          }}
        >
          AI Recruitment System
        </Typography>
        <Box>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
          >
            Dashboard
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/upload-job"
          >
            Upload Job
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/upload-candidate"
          >
            Upload Candidate
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/matches"
          >
            Matches
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 