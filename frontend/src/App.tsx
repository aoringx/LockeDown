import { useState } from 'react'
import './App.css'
import { Box } from '@mui/material'
import Header from './components/Header'
import HomePage from './components/Home'
import DashboardPage from './components/Dashboard'
import LeaderboardPage from './components/Leaderboard'
import Footer from './components/Footer'

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'dashboard' | 'leaderboard'>('home')

  const handleNavigate = (page: string) => {
  setCurrentPage(page as 'home' | 'dashboard' | 'leaderboard')
  }

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage onNavigate={handleNavigate} />
      case 'dashboard':
        return <DashboardPage onNavigate={handleNavigate} />
      case 'leaderboard':
        return <LeaderboardPage onNavigate={handleNavigate} />
      default:
        return <HomePage onNavigate={handleNavigate} />
    }
  }

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />
      
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', width: '100%' }}>
        {renderCurrentPage()}
      </Box>

      <Footer />
    </Box>
  )
}

export default App
