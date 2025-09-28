import { Box, Typography, Fab } from '@mui/material'


type HomePageProps = {
  onNavigate?: (page: string) => void
}

export default function HomePage({ onNavigate }: HomePageProps) {
  return (
    <Box sx={{ flex: 1, width: '100%', position: 'fixed', p: 2 }}>

      <Fab
        color="primary"
        aria-label="open dashboard"
        onClick={() => onNavigate?.('dashboard')}
        sx={{
          position: 'fixed',
          top: '40%',
          left: '40%',
          transform: 'translate(-50%, -50%)',
          width: 400,
          height: 400,
          boxShadow: 6
        }}
      >
        <Typography
          variant="h2"
          component="span"
          sx={{ fontSize: { xs: 36, sm: 48, md: 50 }, fontWeight: 500 }}
        >
          Get Started
        </Typography>
      </Fab>

      <Fab
        color="secondary"
        aria-label="open leaderboard"
        onClick={() => onNavigate?.('leaderboard')}
        sx={{
          position: 'fixed',
          top: '70%',
          left: '70%',
          transform: 'translate(-50%, -50%)',
          width: 200,
          height: 200,
          boxShadow: 6
        }}
      >
        <Typography variant="h6" component="span" sx={{ mr: 1 }}>
          Leaderboard
        </Typography>
      </Fab>
    </Box>
  )
}