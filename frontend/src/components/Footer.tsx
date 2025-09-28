import { Box, Paper, Typography } from '@mui/material'

function Footer() {
  const year = new Date().getFullYear()
  return (
    <Paper component="footer" square variant="outlined">
      <Box >
        <Typography variant="body2" color="text.secondary" align="center">
          &copy; {year} Lockedown Website.
        </Typography>
      </Box>
    </Paper>
  )
}

export default Footer
