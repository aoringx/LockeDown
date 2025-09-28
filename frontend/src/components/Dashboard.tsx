import { Box, Typography, Fab, Paper, Table, TableHead, TableRow, TableCell, TableBody, TableContainer } from '@mui/material'
import { useEffect, useState } from 'react'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import LeaderboardIcon from '@mui/icons-material/Leaderboard'

type SessionCurrent = {
  category: string
  reason: string
  last_data: string
  total_entertainment_time: number
  points: number
}

type HistoryEvent = {
  timestamp: string
  category: string
  data: string
  reason: string
  total_time_min: number
  content?: string
}

type SessionResponse = {
  current: SessionCurrent
  history: HistoryEvent[]
}

type DashboardProps = {
  onNavigate?: (page: string) => void
}

function DashboardPage({ onNavigate }: DashboardProps) {
  const [sessionData, setSessionData] = useState<SessionResponse | null>(null)
  const [postureState, setPostureState] = useState<string | null>(null)

  // Automatically sort history by timestamp (most recent first)
  const sortedHistory = sessionData?.history.slice().sort((a, b) => {
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  })

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/session');
        const data: SessionResponse = await response.json();
        setSessionData(data);
      } catch (err) {
        console.error('Failed to fetch data:', err);
      }
    };

    const fetchPosture = async () => {
      try {
        const response = await fetch('http://localhost:5000/data');
        const data = await response.json();
        console.log('Current state:', data.state);
        setPostureState(data.state);
      } catch (err) {
        console.error('Failed to fetch posture data:', err);
      }
    };

    // Fetch immediately
    fetchData();
    fetchPosture();

    // Then fetch every 5 seconds
    const interval = setInterval(() => {
      fetchData();
      fetchPosture();
    }, 2000);

    return () => clearInterval(interval);
  }, []);


  
  return (
    <Box className="dashboard" sx={{ 
      flex: 1, 
      width: '100%', 
      mt: 4, 
      mb: 4, 
      px: 2, 
      minHeight: '100vh',
      position: 'relative',
      paddingBottom: '100px'
    }}>
      <Typography variant="h2" component="h1" align="center">
        Dashboard
      </Typography>
      <Typography
        variant="h4"
        align="center"
        sx={{ color: postureState && postureState !== 'Good' ? 'error.main' : 'text.secondary' }}
      >
        Your Posture: {postureState || 'Loading...'}
      </Typography>
      <Box>
        <Typography variant="h4" align="center">Current Session</Typography>
        <Paper elevation={2} sx={{ maxWidth: 900, mx: 'auto', p: 2, mb: 4 }}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: '260px 1fr' },
              alignItems: 'center',
              rowGap: 1.5,
              columnGap: 2,
            }}
          >
            <Typography variant="subtitle2">Category:</Typography>
            <Box component="span">{sessionData?.current.category || 'Loading...'}</Box>

            <Typography variant="subtitle2">Reason:</Typography>
            <Box component="span">{sessionData?.current.reason || 'Loading...'}</Box>

            <Typography variant="subtitle2">Last Data:</Typography>
            <Box component="span">{sessionData?.current.last_data || 'Loading...'}</Box>

            <Typography variant="subtitle2">Total Entertainment Time (min):</Typography>
            <Box component="span">{sessionData ? (sessionData.current.total_entertainment_time / 60).toFixed(2) : 'Loading...'}</Box>

            <Typography variant="subtitle2">Current Points:</Typography>
            <Box component="span">{sessionData ? sessionData.current.points.toFixed(2) : 'Loading...'}</Box>
          </Box>
        </Paper>

        <Typography variant="h4" align="center">Session History</Typography>
        <TableContainer component={Paper} sx={{ maxWidth: 900, mx: 'auto' }}>
          <Table id="history" size="small" aria-label="session history">
            <TableHead>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Behavior</TableCell>
                <TableCell>Total Time (min)</TableCell>
                <TableCell>Details</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedHistory?.map((event, index) => {
                const eventDate = new Date(event.timestamp)
                const now = new Date()
                const oneDay = 24 * 60 * 60 * 1000
                const diffDays = Math.floor((Number(now) - Number(eventDate)) / oneDay)
                
                let dayString: string
                if (diffDays === 0) dayString = 'Today'
                else if (diffDays === 1) dayString = 'Yesterday'
                else dayString = eventDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })

                const timeString = eventDate.toLocaleTimeString('en-US', {
                  hour: 'numeric',
                  minute: '2-digit',
                  hour12: true,
                })

                const friendlyTime = `${dayString}, ${timeString}`

                return (
                  <TableRow key={index}>
                    <TableCell>{friendlyTime}</TableCell>
                    <TableCell>{event.category}</TableCell>
                    <TableCell>{event.total_time_min.toFixed(2)}</TableCell>
                    <TableCell>
                      <div>
                        <strong>Data:</strong> {event.data}<br />
                        {event.content && <><strong>Content:</strong> {event.content}<br /></>}
                        <strong>Reason:</strong> {event.reason}
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </TableContainer>


      </Box>
      
      <Fab 
        color="default" 
        aria-label="exit"
        onClick={() => onNavigate?.('home')}
        sx={{ position: 'fixed', right: 24, bottom: 24 }}
      >
        <ArrowBackIcon />
      </Fab>

      <Fab 
        color="secondary" 
        aria-label="leaderboard"
        onClick={() => onNavigate?.('leaderboard')}
        sx={{ position: 'fixed', left: 24, bottom: 24 }}
      >
        <LeaderboardIcon />
      </Fab>
      
    </Box>
  )
}

export default DashboardPage
