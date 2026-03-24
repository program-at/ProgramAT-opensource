import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  TextInput,
  StyleSheet,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  AccessibilityInfo,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import BeepService from './BeepService';
import WebSocketService from './WebSocketService';
import { useTheme } from './ThemeContext';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'image';
  content: string;
  timestamp: Date;
  imageUri?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  conversationId?: string; // Server-side conversation ID for follow-up questions
  imageUri?: string; // The image associated with this conversation (for display)
  toolName?: string; // Name of the tool used to create this conversation
  createdAt: Date;
  lastUpdated: Date;
}

interface ChatProps {
  webSocketService?: any;
  initialConversationId?: string; // Open a specific conversation on mount
}

const Chat: React.FC<ChatProps> = ({ webSocketService, initialConversationId }) => {
  const { theme } = useTheme();
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);
  const hasOpenedInitialConversation = useRef(false); // Track if we've already opened the initial conversation

  useEffect(() => {
    loadChatSessions();
    
    // Poll for new chats every 2 seconds
    const interval = setInterval(() => {
      console.log('[Chat] Polling for new chat sessions');
      loadChatSessions();
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  // Open specific conversation when initialConversationId is provided (only once)
  useEffect(() => {
    if (initialConversationId && chatSessions.length > 0 && !hasOpenedInitialConversation.current) {
      const matchingSession = chatSessions.find(
        session => session.conversationId === initialConversationId
      );
      if (matchingSession) {
        console.log('[Chat] Opening conversation:', initialConversationId);
        setActiveChatId(matchingSession.id);
        hasOpenedInitialConversation.current = true; // Mark as opened so we don't open again
      }
    }
  }, [initialConversationId, chatSessions]);

  useEffect(() => {
    // Scroll to bottom when new messages are added
    if (scrollViewRef.current) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [activeChatId, chatSessions]);

  // Handle loading beeps
  useEffect(() => {
    let beepTimer: ReturnType<typeof setTimeout> | null = null;
    
    if (isLoading) {
      console.log('[Chat] Processing message, will beep after 3 seconds if still loading');
      // Wait 3 seconds before starting beep
      beepTimer = setTimeout(() => {
        console.log('[Chat] 3 seconds elapsed, starting loading sound');
        BeepService.startLoadingSound();
      }, 3000);
    } else {
      console.log('[Chat] Stopping loading beeps');
      BeepService.stopLoadingSound();
    }

    // Cleanup on unmount or when loading state changes
    return () => {
      if (beepTimer) {
        clearTimeout(beepTimer);
      }
      BeepService.stopLoadingSound();
    };
  }, [isLoading]);

  // Announce initial message when opening a chat
  useEffect(() => {
    if (activeChatId) {
      const activeChat = chatSessions.find(chat => chat.id === activeChatId);
      if (activeChat && activeChat.messages.length > 0) {
        // Find the first assistant message (initial response)
        const firstAssistantMessage = activeChat.messages.find(msg => msg.type === 'assistant');
        if (firstAssistantMessage) {
          console.log('[Chat] Announcing initial response');
          // Small delay to let the screen transition complete
          setTimeout(() => {
            AccessibilityInfo.announceForAccessibility(firstAssistantMessage.content);
          }, 500);
        }
      }
    }
  }, [activeChatId]);

  const loadChatSessions = () => {
    const storageKey = 'chatSessions';
    console.log('[Chat] Loading chat sessions from AsyncStorage');
    AsyncStorage.getItem(storageKey)
      .then((sessionsData) => {
        if (sessionsData) {
          console.log('[Chat] Raw sessions data length:', sessionsData.length);
          const sessions = JSON.parse(sessionsData).map((session: any) => ({
            ...session,
            createdAt: new Date(session.createdAt),
            lastUpdated: new Date(session.lastUpdated),
            messages: session.messages.map((msg: any) => ({
              ...msg,
              timestamp: new Date(msg.timestamp),
            })),
          }));
          console.log('[Chat] Loaded', sessions.length, 'sessions');
          sessions.forEach((s: any, i: number) => {
            console.log(`[Chat] Session ${i}: ${s.id}, messages: ${s.messages.length}`);
            s.messages.forEach((m: any, j: number) => {
              console.log(`[Chat]   Message ${j}: type=${m.type}, hasImage=${!!m.imageUri}, imageUriLength=${m.imageUri?.length || 0}`);
            });
          });
          setChatSessions(sessions.sort((a: any, b: any) => b.lastUpdated.getTime() - a.lastUpdated.getTime()));
        } else {
          console.log('[Chat] No sessions data found');
          setChatSessions([]);
        }
      })
      .catch((error) => {
        console.error('Error loading chat sessions:', error);
      });
  };

  const saveChatSessions = (sessions: ChatSession[]) => {
    const storageKey = 'chatSessions';
    console.log('[Chat] Saving', sessions.length, 'chat sessions');
    AsyncStorage.setItem(storageKey, JSON.stringify(sessions))
      .catch((error) => {
        console.error('Error saving chat sessions:', error);
      });
  };

  const createNewChat = (imageUri?: string, initialMessage?: string, toolName?: string) => {
    const newChatId = Date.now().toString();
    const now = new Date();
    
    // Create descriptive title: "Tool Name - Date at Time"
    const dateStr = now.toLocaleDateString();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const title = toolName 
      ? `${toolName} - ${dateStr} at ${timeStr}`
      : `Chat ${dateStr} at ${timeStr}`;
    
    const newChat: ChatSession = {
      id: newChatId,
      title,
      toolName,
      messages: [],
      createdAt: new Date(),
      lastUpdated: new Date(),
    };

    if (imageUri) {
      const imageMessage: ChatMessage = {
        id: `${Date.now()}_img`,
        type: 'image',
        content: 'Image captured',
        timestamp: new Date(),
        imageUri,
      };
      newChat.messages.push(imageMessage);
    }

    if (initialMessage) {
      const assistantMessage: ChatMessage = {
        id: `${Date.now()}_assistant`,
        type: 'assistant',
        content: initialMessage,
        timestamp: new Date(),
      };
      newChat.messages.push(assistantMessage);
    }

    const updatedSessions = [newChat, ...chatSessions];
    setChatSessions(updatedSessions);
    setActiveChatId(newChatId);
    saveChatSessions(updatedSessions);

    return newChatId;
  };

  const addMessageToChat = (chatId: string, message: ChatMessage) => {
    console.log('[Chat] addMessageToChat called for chat:', chatId, 'message type:', message.type);
    
    // Use functional update to ensure we're working with latest state
    setChatSessions(prevSessions => {
      const updatedSessions = prevSessions.map(session => {
        if (session.id === chatId) {
          const updatedSession = {
            ...session,
            messages: [...session.messages, message],
            lastUpdated: new Date(),
          };
          console.log('[Chat] Updated session now has', updatedSession.messages.length, 'messages');
          console.log('[Chat] Message types:', updatedSession.messages.map(m => m.type).join(', '));
          return updatedSession;
        }
        return session;
      });
      
      // Save to AsyncStorage with the updated sessions
      saveChatSessions(updatedSessions);
      return updatedSessions;
    });
  };

  const sendMessage = () => {
    if (!inputText.trim() || !activeChatId || !webSocketService) return;

    const userMessage: ChatMessage = {
      id: `${Date.now()}_user`,
      type: 'user',
      content: inputText.trim(),
      timestamp: new Date(),
    };

    console.log('[Chat] Adding user message to chat:', userMessage);
    addMessageToChat(activeChatId, userMessage);
    setInputText('');
    setIsLoading(true);

    // Get the conversation ID from the active chat
    const activeChat = chatSessions.find(chat => chat.id === activeChatId);
    const conversationId = activeChat?.conversationId;

    console.log('[Chat] Sending follow-up question for conversation:', conversationId);

    if (conversationId) {
      // Send the follow-up question with the conversation ID (image is stored on server)
      webSocketService.sendFollowUpQuestion(
        conversationId,
        inputText.trim()
      ).then((response: string | null) => {
        if (response) {
          const assistantMessage: ChatMessage = {
            id: `${Date.now()}_assistant`,
            type: 'assistant',
            content: response,
            timestamp: new Date(),
          };
          addMessageToChat(activeChatId, assistantMessage);
          
          // Announce the response for VoiceOver/TalkBack
          console.log('[Chat] Announcing assistant response via accessibility');
          AccessibilityInfo.announceForAccessibility(response);
        }
        setIsLoading(false);
      }).catch((error: any) => {
        console.error('Error sending message:', error);
        setIsLoading(false);
      });
    } else {
      // No conversation ID found
      console.warn('[Chat] No conversation ID found for this chat');
      const assistantMessage: ChatMessage = {
        id: `${Date.now()}_assistant`,
        type: 'assistant',
        content: 'This conversation does not have an associated image. Please start a new conversation.',
        timestamp: new Date(),
      };
      addMessageToChat(activeChatId, assistantMessage);
      setIsLoading(false);
    }
  };

  const deleteChat = (chatId: string) => {
    Alert.alert(
      'Delete Chat',
      'Are you sure you want to delete this chat?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            const updatedSessions = chatSessions.filter(session => session.id !== chatId);
            setChatSessions(updatedSessions);
            saveChatSessions(updatedSessions);
            if (activeChatId === chatId) {
              setActiveChatId(null);
            }
          },
        },
      ]
    );
  };

  const renderMessage = (message: ChatMessage) => {
    // Special handling for image messages
    if (message.type === 'image' && message.imageUri) {
      console.log('[Chat] Rendering image message, URI length:', message.imageUri.length);
      return (
        <View key={message.id} style={[styles.imageMessageContainer, { backgroundColor: theme.card, borderColor: theme.border }]}>
          <Image 
            source={{ uri: message.imageUri }} 
            style={styles.messageImage}
            resizeMode="cover"
            onError={(e) => {
              console.log('[Chat] Image load error:', e.nativeEvent.error);
              if (message.imageUri) {
                console.log('[Chat] Failed image URI prefix:', message.imageUri.substring(0, 100));
              }
            }}
            onLoad={() => console.log('[Chat] Image loaded successfully')}
          />
          <Text style={[styles.timestamp, { color: theme.textTertiary }]}>
            {message.timestamp.toLocaleTimeString()}
          </Text>
        </View>
      );
    }
    
    // Regular user/assistant messages
    return (
      <TouchableOpacity
        key={message.id}
        activeOpacity={1}
        style={[
          styles.messageContainer,
          message.type === 'user' ? { backgroundColor: theme.primary } : { backgroundColor: theme.card, borderColor: theme.border }
        ]}
        accessible={true}
        accessibilityLabel={`${message.type === 'user' ? 'You said' : 'Assistant said'}: ${message.content}`}
        accessibilityRole="text"
        accessibilityHint="Long press to copy text"
      >
        <Text 
          style={[
            styles.messageText,
            message.type === 'user' ? styles.userMessageText : { color: theme.text }
          ]}
          selectable={true}
          accessible={false}>
          {message.content}
        </Text>
        <Text 
          style={[styles.timestamp, { color: theme.textTertiary }]}
          selectable={false}
          accessible={false}>
          {message.timestamp.toLocaleTimeString()}
        </Text>
      </TouchableOpacity>
    );
  };

  const activeChat = chatSessions.find(chat => chat.id === activeChatId);

  if (!activeChatId || !activeChat) {
    return (
      <View style={[styles.container, { backgroundColor: theme.background }]}>
        <Text style={[styles.title, { color: theme.text }]}>Chat History</Text>
        <ScrollView style={styles.chatList}>
          {chatSessions.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={[styles.emptyText, { color: theme.text }]}>No conversations yet</Text>
              <Text style={[styles.emptySubtext, { color: theme.textSecondary }]}>
                Start a conversation by using the conversation mode in the Tools tab
              </Text>
            </View>
          ) : (
            chatSessions.map(session => (
              <TouchableOpacity
                key={session.id}
                style={[styles.chatSessionItem, { backgroundColor: theme.backgroundSecondary, borderColor: theme.border }]}
                onPress={() => setActiveChatId(session.id)}
              >
                <View style={styles.chatSessionHeader}>
                  <Text style={[styles.chatSessionTitle, { color: theme.text }]}>{session.title}</Text>
                  <TouchableOpacity
                    onPress={() => deleteChat(session.id)}
                    style={[styles.deleteButton, { backgroundColor: theme.error }]}
                  >
                    <Text style={styles.deleteButtonText}>×</Text>
                  </TouchableOpacity>
                </View>
                <Text style={[styles.chatSessionDate, { color: theme.textTertiary }]}>
                  {session.lastUpdated.toLocaleDateString()} {session.lastUpdated.toLocaleTimeString()}
                </Text>
                <Text style={[styles.chatSessionPreview, { color: theme.textSecondary }]} numberOfLines={2}>
                  {session.messages[session.messages.length - 1]?.content || 'No messages'}
                </Text>
              </TouchableOpacity>
            ))
          )}
        </ScrollView>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView 
      style={[styles.container, { backgroundColor: theme.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={[styles.header, { backgroundColor: theme.background, borderBottomColor: theme.border }]}>
        <TouchableOpacity
          onPress={() => setActiveChatId(null)}
          style={styles.backButton}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel="All chats"
          accessibilityHint="Goes back to the chat list"
        >
          <Text style={[styles.backButtonText, { color: theme.primary }]}>← All Chats</Text>
        </TouchableOpacity>
        <Text style={[styles.chatTitle, { color: theme.text }]}>{activeChat.title}</Text>
        <TouchableOpacity
          onPress={() => deleteChat(activeChatId)}
          style={[styles.deleteButton, { backgroundColor: theme.error }]}
        >
          <Text style={styles.deleteButtonText}>×</Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        ref={scrollViewRef}
        style={[styles.messagesContainer, { backgroundColor: theme.backgroundSecondary }]}
        contentContainerStyle={styles.messagesContent}
        accessible={true}
        accessibilityLabel="Chat messages"
        accessibilityLiveRegion="polite"
      >
        {(() => {
          console.log('[Chat] Rendering', activeChat.messages.length, 'messages');
          activeChat.messages.forEach((msg, idx) => {
            console.log(`[Chat] Message ${idx}: type=${msg.type}, content=${msg.content.substring(0, 50)}`);
          });
          return activeChat.messages.map(renderMessage);
        })()}
        {isLoading && (
          <View style={[styles.loadingMessage, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Text style={[styles.loadingText, { color: theme.textSecondary }]}>Thinking...</Text>
          </View>
        )}
      </ScrollView>

      <View style={[styles.inputContainer, { backgroundColor: theme.background, borderTopColor: theme.border }]}>
        <TextInput
          style={[styles.textInput, { 
            backgroundColor: theme.inputBackground,
            borderColor: theme.inputBorder,
            color: theme.text
          }]}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Ask a follow-up question..."
          placeholderTextColor={theme.inputPlaceholder}
          multiline
          maxLength={500}
          editable={!isLoading}
          accessible={true}
          accessibilityLabel="Message input"
          accessibilityHint="Type your follow-up question here. Use context menu to copy or paste"
          contextMenuHidden={false}
          selectTextOnFocus={false}
          selection={undefined}
        />
        <TouchableOpacity
          style={[
            styles.sendButton, 
            { backgroundColor: theme.primary },
            (!inputText.trim() || isLoading) && styles.sendButtonDisabled
          ]}
          onPress={sendMessage}
          disabled={!inputText.trim() || isLoading}
          accessible={true}
          accessibilityLabel={isLoading ? "Sending message" : "Send message"}
          accessibilityHint="Sends your question to the assistant"
          accessibilityRole="button"
          accessibilityState={{ disabled: !inputText.trim() || isLoading }}
        >
          <Text style={styles.sendButtonText}>Send</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    paddingVertical: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 15,
    borderBottomWidth: 1,
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    fontSize: 16,
  },
  chatTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'center',
  },
  deleteButton: {
    padding: 5,
    borderRadius: 15,
    width: 30,
    height: 30,
    alignItems: 'center',
    justifyContent: 'center',
  },
  deleteButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  chatList: {
    flex: 1,
    padding: 15,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  chatSessionItem: {
    padding: 15,
    marginBottom: 10,
    borderRadius: 10,
    borderWidth: 1,
  },
  chatSessionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  chatSessionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    flex: 1,
  },
  chatSessionDate: {
    fontSize: 12,
    marginBottom: 5,
  },
  chatSessionPreview: {
    fontSize: 14,
    lineHeight: 18,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 15,
  },
  conversationImageContainer: {
    marginBottom: 20,
    padding: 10,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 1,
  },
  conversationImageLabel: {
    fontSize: 12,
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  conversationImage: {
    width: '100%',
    height: 200,
    borderRadius: 8,
  },
  messageContainer: {
    marginBottom: 15,
    padding: 10,
    borderRadius: 10,
    maxWidth: '85%',
  },
  userMessage: {
    alignSelf: 'flex-end',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    borderWidth: 1,
  },
  imageMessageContainer: {
    marginBottom: 15,
    padding: 10,
    borderRadius: 10,
    alignSelf: 'flex-start',
    maxWidth: '85%',
    borderWidth: 1,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 20,
  },
  userMessageText: {
    color: 'white',
  },
  assistantMessageText: {
  },
  messageImage: {
    width: 200,
    height: 150,
    borderRadius: 8,
    marginBottom: 8,
  },
  timestamp: {
    fontSize: 12,
    marginTop: 5,
  },
  loadingMessage: {
    padding: 10,
    borderRadius: 10,
    alignSelf: 'flex-start',
    marginBottom: 15,
    borderWidth: 1,
  },
  loadingText: {
    fontStyle: 'italic',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 15,
    alignItems: 'flex-end',
    borderTopWidth: 1,
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginRight: 10,
    maxHeight: 100,
    fontSize: 16,
  },
  sendButton: {
    borderRadius: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  sendButtonDisabled: {
    opacity: 0.5,
  },
  sendButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
});

// Export function to create new chat from external components
export const createNewChatSession = (imageUri?: string, initialMessage?: string, toolName?: string) => {
  // This will be called from ToolRunner when conversation mode is used
  return {
    imageUri,
    initialMessage,
    toolName,
    timestamp: new Date(),
  };
};

export default Chat;
