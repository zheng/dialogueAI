import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

// NOTE: intentionally NOT wrapped in <React.StrictMode>. StrictMode double-
// invokes effects in dev (mount → unmount → mount), which creates/destroys/
// recreates the Silero audio-worklet VAD (and the WebSocket) in rapid
// succession. The worklet doesn't survive that churn and ends up never
// listening. One clean mount keeps the mic pipeline stable.
ReactDOM.createRoot(document.getElementById('root')).render(<App />)
