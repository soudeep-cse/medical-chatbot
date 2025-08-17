import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import Skeleton from './components/Skeleton';

// Define the message type
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'ai';
  source?: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [userId, setUserId] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [sourceExpanded, setSourceExpanded] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Simulate initial loading
    setTimeout(() => {
      setIsLoading(false);
    }, 1500);

    // Generate or retrieve user ID
    let storedUserId = localStorage.getItem('userId');
    if (!storedUserId) {
      storedUserId = uuidv4();
      localStorage.setItem('userId', storedUserId);
    }
    setUserId(storedUserId);
  }, []);

  const handleNewChat = () => {
    const newUserId = uuidv4();
    localStorage.setItem('userId', newUserId);
    setUserId(newUserId);
    setMessages([]);
  };

  const parseResponse = (responseText: string): { text: string; source?: string } => {
    const sourceMatch = responseText.match(/\(Source: (.*?)\)/);
    if (sourceMatch) {
      const text = responseText.replace(sourceMatch[0], '').trim();
      return { text, source: sourceMatch[1] };
    }
    return { text: responseText };
  };

  const handleSendMessage = async () => {
    if (inputText.trim() === '' || !userId) return;

    const userMessage: Message = {
      id: messages.length + 1,
      text: inputText,
      sender: 'user',
    };

    setMessages([...messages, userMessage]);
    setInputText('');
    setIsTyping(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        user_id: userId,
        question: inputText,
      });

      const { text, source } = parseResponse(response.data.response);

      const aiMessage: Message = {
        id: messages.length + 2,
        text,
        sender: 'ai',
        source,
      };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: messages.length + 2,
        text: 'Sorry, something went wrong. Please try again.',
        sender: 'ai',
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className={`flex h-screen ${isDarkMode ? 'dark' : ''} bg-gray-100 dark:bg-gray-900 font-sans`}>
      {/* Sidebar */}
      <div className="w-64 bg-white dark:bg-gray-800 p-4 border-r border-gray-200 dark:border-gray-700 flex flex-col">
        <h1 className="text-2xl font-bold text-blue-600 dark:text-blue-400">DocBot</h1>
        <button
          onClick={handleNewChat}
          className="w-full mt-6 bg-blue-500 hover:bg-blue-600 text-white p-2 rounded-md transition-colors"
        >
          + New Chat
        </button>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white dark:bg-gray-800 p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <h2 className="text-xl font-semibold">Your Personal Medical Assistant</h2>
          <div className="flex items-center">
            <button className="mr-4 text-gray-600 dark:text-gray-300">🌐 EN</button>
            <button onClick={toggleDarkMode} className="text-gray-600 dark:text-gray-300">
              {isDarkMode ? '🌙' : '☀️'}
            </button>
          </div>
        </header>

        {/* Message List */}
        <div className="flex-1 p-6 overflow-y-auto">
          {isLoading ? (
            <>
              <Skeleton />
              <Skeleton />
            </>
          ) : (
            messages.map((message) => (
              <div key={message.id} className="animate-fade-in-up">
                <div
                  className={`flex items-end mb-2 ${
                    message.sender === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  {message.sender === 'ai' && (
                    <div className="w-10 h-10 rounded-full bg-green-500 mr-3 flex-shrink-0 flex items-center justify-center text-white font-bold">
                      Dr
                    </div>
                  )}
                  <div
                    className={`max-w-xl p-4 rounded-2xl shadow-md ${
                      message.sender === 'user'
                        ? 'bg-blue-500 text-white rounded-br-none'
                        : 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-bl-none'
                    }`}
                  >
                    {message.text}
                  </div>
                </div>
                {message.source && (
                  <div className="flex justify-start ml-14 mb-4">
                    <div className="w-full max-w-xl">
                      <button
                        onClick={() => setSourceExpanded(sourceExpanded === message.id ? null : message.id)}
                        className="text-xs text-blue-500 dark:text-blue-400 focus:outline-none"
                      >
                        {sourceExpanded === message.id ? 'Hide Source' : 'Show Source'}
                      </button>
                      {sourceExpanded === message.id && (
                        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 p-2 bg-gray-100 dark:bg-gray-800 rounded-md">
                          <strong>Source:</strong> {message.source}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
          {isTyping && <div className="animate-fade-in-up"><Skeleton /></div>}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <div className="p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your symptoms or question here..."
              className="flex-1 p-3 border border-gray-300 rounded-full px-6 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleSendMessage}
              className="ml-3 bg-green-500 hover:bg-green-600 text-white p-3 rounded-full flex items-center justify-center transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
