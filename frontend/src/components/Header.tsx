import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button
} from '@mui/material'
import { Home, Dashboard } from '@mui/icons-material'


function Header() {
  return (
    <>
      {/* Header with MUI AppBar */}
      <AppBar position="fixed">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            LockeDown
          </Typography>
          
        </Toolbar>
      </AppBar>

      {/* Toolbar spacer to prevent content overlap */}
      <Toolbar />
    </>
  )
}

export default Header