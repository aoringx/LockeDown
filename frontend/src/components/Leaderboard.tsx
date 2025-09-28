import { Box, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Fab } from '@mui/material'
import { useEffect, useState } from 'react'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'

export type LeaderboardEntry = {
  rank: number
  name: string
  score: number
  isCurrentUser?: boolean
}

type LeaderboardProps = {
  onNavigate?: (page: string) => void
  data?: LeaderboardEntry[]
}

const defaultData: LeaderboardEntry[] = [
  { rank: 1, name: 'Alice', score: 120 },
  { rank: 2, name: 'Bob', score: 121 },
  { rank: 3, name: 'Charlie', score: 167 },
  { rank: 4, name: 'Diana', score: 93 },
  { rank: 5, name: 'Eve', score: 56 }
]

export default function LeaderboardPage({ onNavigate, data = defaultData }: LeaderboardProps) {
  const [userScore, setUserScore] = useState<number | null>(null)
  const [leaderboardData, setLeaderboardData] = useState<LeaderboardEntry[]>(data)

  useEffect(() => {
    const fetchUserScore = async () => {
      try {
  const response = await fetch('http://localhost:8000/session')
        const sessionData = await response.json()
        const score = sessionData.current.points
        setUserScore(score)

        // Add user to leaderboard data and sort by score (highest first)
        const updatedData = [
          ...data.filter(entry => entry.name !== 'You'), // Remove existing user entry if any
          { rank: 0, name: 'You', score: score, isCurrentUser: true }
        ].sort((a, b) => b.score - a.score)

        // Update ranks based on sorted order
        const rankedData = updatedData.map((entry, index) => ({
          ...entry,
          rank: index + 1
        }))

        setLeaderboardData(rankedData)
      } catch (err) {
        console.error('Failed to fetch user score:', err)
        setLeaderboardData(data) // Fall back to default data
      }
    }

    fetchUserScore()
    // Update every 10 seconds
    const interval = setInterval(fetchUserScore, 10000)
    return () => clearInterval(interval)
  }, [data])
  return (
    <Box sx={{ 
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      width: '100%', 
      maxWidth: '1200px',
      margin: '0 auto',
      mt: 4, 
      mb: 4, 
      px: 2, 
      minHeight: '100vh',
      paddingBottom: '100px'
    }}>
      <Typography variant="h2" component="h1" gutterBottom align="center">
        Leaderboard
      </Typography>

      <TableContainer component={Paper} sx={{ maxWidth: 900, mx: 'auto' }}>
        <Table size="medium" aria-label="leaderboard">
          <TableHead>
            <TableRow>
              <TableCell align="center">Rank</TableCell>
              <TableCell>Name</TableCell>
              <TableCell align="right">Score</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {leaderboardData.map((row) => (
              <TableRow 
                key={`${row.name}-${row.rank}`} 
                sx={{
                  backgroundColor: row.isCurrentUser ? 'primary.light' : 'inherit'
                }}
              >
                <TableCell align="center" sx={{ fontWeight: row.isCurrentUser ? 'bold' : 'normal' }}>
                  {row.rank}
                </TableCell>
                <TableCell sx={{ fontWeight: row.isCurrentUser ? 'bold' : 'normal' }}>
                  {row.name}
                </TableCell>
                <TableCell align="right" sx={{ fontWeight: row.isCurrentUser ? 'bold' : 'normal' }}>
                  {Math.round(row.score)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Back button bottom-right */}
      <Fab 
        color="default" 
        aria-label="back"
        onClick={() => onNavigate?.('home')}
        sx={{ position: 'fixed', right: 24, bottom: 24 }}
      >
        <ArrowBackIcon />
      </Fab>
    </Box>
  )
}
