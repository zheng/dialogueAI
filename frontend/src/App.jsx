import { useState } from 'react'
import Lobby from './Lobby.jsx'
import ChatView from './ChatView.jsx'

// Tiny state machine: the lobby until an expert is matched, then the chat.
export default function App() {
  const [expertId, setExpertId] = useState(null)

  return expertId ? (
    <ChatView expertId={expertId} onBack={() => setExpertId(null)} />
  ) : (
    <Lobby onMatch={setExpertId} />
  )
}
