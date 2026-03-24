/**
 * WebSocketService
 * Manages WebSocket connection to the backend stream server
 * Handles sending frames and text transcriptions
 *
 * @format
 */

import Config from './config';

export interface StreamFrame {
  frameNumber: number;
  timestamp: number;
  userId?: number; // Optional for backward compatibility
  data: {
    base64Image: string;
    width?: number;
    height?: number;
  };
}

export interface StreamText {
  text: string;
  timestamp: number;
  userId?: number; // Optional for backward compatibility
}

export interface ServerMessage {
  type: string;
  [key: string]: any;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private serverUrl: string = Config.WEBSOCKET_SERVER_URL;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = Config.MAX_RECONNECT_ATTEMPTS;
  private reconnectDelay: number = Config.RECONNECT_DELAY_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isConnecting: boolean = false;
  private frameNumber: number = 0;
  private connectionTimeout: ReturnType<typeof setTimeout> | null = null;
  private readonly CONNECTION_TIMEOUT_MS = 15000; // 15 second timeout
  
  // Heartbeat mechanism
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private heartbeatTimeoutTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly HEARTBEAT_INTERVAL_MS = 25000; // Send ping every 25 seconds (between server's 20s pings)
  private readonly HEARTBEAT_TIMEOUT_MS = 8000; // Wait 8 seconds for pong response (less than server's 10s timeout)
  private missedHeartbeats: number = 0;
  private readonly MAX_MISSED_HEARTBEATS = 2; // Reconnect after 2 missed pongs (faster detection)
  
  // Callbacks
  private onConnectionChangeCallback?: (connected: boolean) => void;
  private onMessageCallback?: (message: ServerMessage) => void;
  private onErrorCallback?: (error: string) => void;

  constructor(serverUrl?: string) {
    if (serverUrl) {
      this.serverUrl = serverUrl;
    }
  }

  /**
   * Get the current server URL
   */
  getServerUrl(): string {
    return this.serverUrl;
  }

  /**
   * Set a new server URL and optionally reconnect
   */
  setServerUrl(url: string, reconnect: boolean = false): void {
    console.log(`[WebSocketService] setServerUrl called`);
    console.log(`[WebSocketService] Old URL: ${this.serverUrl}`);
    console.log(`[WebSocketService] New URL: ${url}`);
    this.serverUrl = url;
    console.log(`[WebSocketService] URL now set to: ${this.serverUrl}`);
    if (reconnect) {
      console.log(`[WebSocketService] Reconnect requested, disconnecting and reconnecting...`);
      this.disconnect();
      this.connect();
    }
  }

  /**
   * Start the heartbeat mechanism
   */
  private startHeartbeat() {
    this.stopHeartbeat(); // Clear any existing heartbeat
    this.missedHeartbeats = 0;
    
    console.log('Starting WebSocket heartbeat');
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected()) {
        this.sendPing();
      } else {
        this.stopHeartbeat();
      }
    }, this.HEARTBEAT_INTERVAL_MS);
  }

  /**
   * Stop the heartbeat mechanism
   */
  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
    this.missedHeartbeats = 0;
  }

  /**
   * Send a ping message to the server
   */
  private sendPing() {
    if (!this.isConnected()) {
      return;
    }

    try {
      console.log('Sending WebSocket ping');
      const pingMessage = {
        type: 'ping',
        timestamp: Date.now(),
      };
      
      this.ws!.send(JSON.stringify(pingMessage));
      
      // Set timeout for pong response
      this.heartbeatTimeoutTimer = setTimeout(() => {
        this.handleMissedHeartbeat();
      }, this.HEARTBEAT_TIMEOUT_MS);
      
    } catch (error) {
      console.error('Failed to send ping:', error);
      this.handleMissedHeartbeat();
    }
  }

  /**
   * Handle a received pong message
   */
  private handlePong() {
    console.log('Received WebSocket pong');
    this.missedHeartbeats = 0;
    
    // Clear the timeout timer since we received the pong
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
  }

  /**
   * Handle missed heartbeat
   */
  private handleMissedHeartbeat() {
    this.missedHeartbeats++;
    console.warn(`Missed heartbeat ${this.missedHeartbeats}/${this.MAX_MISSED_HEARTBEATS}`);
    
    if (this.heartbeatTimeoutTimer) {
      clearTimeout(this.heartbeatTimeoutTimer);
      this.heartbeatTimeoutTimer = null;
    }
    
    if (this.missedHeartbeats >= this.MAX_MISSED_HEARTBEATS) {
      console.error('Max missed heartbeats reached, connection appears dead');
      this.stopHeartbeat();
      
      // Force close the connection safely
      this.closeConnectionSafely();
      
      if (this.onConnectionChangeCallback) {
        try {
          this.onConnectionChangeCallback(false);
        } catch (error) {
          console.error('Error in connection change callback:', error);
        }
      }
      
      // Attempt reconnection
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.attemptReconnect();
      }
    }
  }

  /**
   * Safely close WebSocket connection with error handling
   */
  private closeConnectionSafely() {
    if (this.ws) {
      try {
        // Only close if connection is in a valid state
        if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
          this.ws.close();
        }
      } catch (error) {
        console.error('Error closing WebSocket connection:', error);
        // Continue cleanup even if close fails
      }
      this.ws = null;
    }
  }

  /**
   * Connect to the WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        console.log('WebSocket already connected');
        resolve();
        return;
      }

      if (this.isConnecting) {
        console.log('WebSocket connection already in progress');
        reject(new Error('Connection already in progress'));
        return;
      }

      // Clean up any existing connection in a bad state
      if (this.ws) {
        console.log('Cleaning up existing WebSocket');
        try {
          this.ws.close();
        } catch {
          // Ignore errors during cleanup
        }
        this.ws = null;
      }
      
      // Clear any pending reconnection timer
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
      
      // Reset reconnection attempts for manual connection
      this.reconnectAttempts = 0;

      this.isConnecting = true;
      console.log(`[WebSocketService] ========================================`);
      console.log(`[WebSocketService] CONNECTING TO: ${this.serverUrl}`);
      console.log(`[WebSocketService] ========================================`);
      console.log(`Connection will timeout after ${this.CONNECTION_TIMEOUT_MS / 1000} seconds if not successful`);

      // Set a connection timeout
      this.connectionTimeout = setTimeout(() => {
        if (this.isConnecting) {
          console.log('Connection timeout after', this.CONNECTION_TIMEOUT_MS / 1000, 'seconds');
          console.log('Failed to connect to:', this.serverUrl);
          this.isConnecting = false;
          if (this.ws) {
            try {
              this.ws.close();
            } catch {
              // Ignore errors during cleanup
            }
            this.ws = null;
          }
          const errorMsg = `Connection timeout. Could not reach server at ${this.serverUrl}. Please verify:\n1. Server is running\n2. URL is correct\n3. Network allows WebSocket connections`;
          if (this.onErrorCallback) {
            this.onErrorCallback(errorMsg);
          }
          reject(new Error(errorMsg));
        }
      }, this.CONNECTION_TIMEOUT_MS);

      try {
        this.ws = new WebSocket(this.serverUrl);
        
        // Store reference for cleanup in error handler
        const currentWs = this.ws;
        let hasResolved = false;

        this.ws.onopen = () => {
          console.log('WebSocket connected successfully');
          
          // Clear connection timeout
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.frameNumber = 0;
          hasResolved = true;
          
          // Start heartbeat mechanism
          this.startHeartbeat();
          
          if (this.onConnectionChangeCallback) {
            this.onConnectionChangeCallback(true);
          }
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: ServerMessage = JSON.parse(event.data);
            console.log('Received message from server:', message.type);
            
            // Handle pong responses
            if (message.type === 'pong') {
              this.handlePong();
              return; // Don't forward pong messages to app callbacks
            }
            
            // Extra logging for production_tools messages
            if (message.type === 'production_tools') {
              console.log('[WebSocketService] Production tools received:', message.tools?.length || 0, 'tools');
              console.log('[WebSocketService] Tools:', message.tools);
            }
            
            if (this.onMessageCallback) {
              this.onMessageCallback(message);
            }
          } catch (error) {
            console.error('Failed to parse server message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error during connection:', error);
          // Don't reject here - let onclose handle it to avoid double-rejection
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket connection closed', event.code, event.reason);
          
          // Stop heartbeat mechanism
          this.stopHeartbeat();
          
          // Clear connection timeout
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          // If we haven't resolved yet, this is a failed connection attempt
          if (!hasResolved) {
            this.isConnecting = false;
            
            // Clean up the failed connection
            if (this.ws === currentWs) {
              this.ws = null;
            }
            
            console.log(`Failed to connect. Close code: ${event.code}, reason: ${event.reason || 'No reason provided'}`);
            const errorMsg = `Failed to connect to ${this.serverUrl}. Error code: ${event.code}. Please verify the server is running and accessible.`;
            if (this.onErrorCallback) {
              this.onErrorCallback(errorMsg);
            }
            reject(new Error(errorMsg));
          } else {
            // This was an established connection that closed
            this.isConnecting = false;
            
            if (this.onConnectionChangeCallback) {
              this.onConnectionChangeCallback(false);
            }
            
            // Only attempt reconnect if connection was not cleanly closed
            if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
              this.attemptReconnect();
            }
          }
        };

      } catch (error) {
        // Clear connection timeout
        if (this.connectionTimeout) {
          clearTimeout(this.connectionTimeout);
          this.connectionTimeout = null;
        }
        
        this.isConnecting = false;
        this.ws = null;
        console.error('Failed to create WebSocket:', error);
        const errorMsg = `Failed to connect: ${error}`;
        if (this.onErrorCallback) {
          this.onErrorCallback(errorMsg);
        }
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect() {
    // Stop heartbeat mechanism
    this.stopHeartbeat();
    
    // Clear any pending reconnection attempts
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    // Clear connection timeout
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }
    
    // Prevent automatic reconnection
    this.reconnectAttempts = this.maxReconnectAttempts;
    
    // Reset connecting flag
    this.isConnecting = false;
    
    // Close and clean up WebSocket
    if (this.ws) {
      console.log('Disconnecting WebSocket');
      this.closeConnectionSafely();
    }
  }

  /**
   * Attempt to reconnect to the server
   */
  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached');
      if (this.onErrorCallback) {
        try {
          this.onErrorCallback('Connection failed: Maximum reconnection attempts exceeded');
        } catch (error) {
          console.error('Error in error callback:', error);
        }
      }
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * this.reconnectAttempts;
    console.log(`Attempting reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((error) => {
        console.error('Reconnection failed:', error);
        // Continue attempting reconnections unless max attempts reached
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          setTimeout(() => this.attemptReconnect(), this.reconnectDelay);
        }
      });
    }, delay);
  }

  /**
   * Send a frame to the server
   */
  sendFrame(base64Image: string, width?: number, height?: number): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot send frame: WebSocket not connected');
      return false;
    }

    try {
      const frame: StreamFrame = {
        frameNumber: this.frameNumber++,
        timestamp: Date.now(),
        data: {
          base64Image,
          width,
          height,
        },
      };

      this.ws!.send(JSON.stringify(frame));
      return true;
    } catch (error) {
      console.error('Failed to send frame:', error);
      return false;
    }
  }

  /**
   * Send text transcription to the server
   */
  sendText(text: string): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot send text: WebSocket not connected');
      return false;
    }

    try {
      const message: StreamText = {
        text,
        timestamp: Date.now(),
      };

      this.ws!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to send text:', error);
      return false;
    }
  }

  /**
   * Send follow-up question for conversation mode
   */
  sendFollowUpQuestion(conversationId: string, question: string): Promise<string | null> {
    if (!this.isConnected()) {
      console.warn('Cannot send follow-up question: WebSocket not connected');
      return Promise.resolve(null);
    }

    return new Promise((resolve) => {
      try {
        const message = {
          type: 'follow_up_question',
          question: question,
          conversation_id: conversationId,  // Send conversation ID instead of image
          timestamp: Date.now(),
        };

        console.log('[WebSocket] Sending follow-up question for conversation:', conversationId);

        // Set up one-time listener for the response
        const handleResponse = (event: any) => {
          try {
            const response = JSON.parse(event.data);
            if (response.type === 'follow_up_response') {
              if (this.ws) {
                this.ws.removeEventListener('message', handleResponse);
              }
              if (response.status === 'success') {
                resolve(response.response || 'No response received');
              } else {
                resolve(`Error: ${response.error || 'Unknown error'}`);
              }
            }
          } catch (error) {
            console.error('Error parsing follow-up response:', error);
            resolve('Error parsing response');
          }
        };

        if (this.ws) {
          this.ws.addEventListener('message', handleResponse);
          
          // Send the question
          this.ws.send(JSON.stringify(message));
          
          // Timeout after 30 seconds
          setTimeout(() => {
            if (this.ws) {
              this.ws.removeEventListener('message', handleResponse);
            }
            resolve('Request timed out');
          }, 30000);
        } else {
          resolve('WebSocket connection lost');
        }

      } catch (error) {
        console.error('Failed to send follow-up question:', error);
        resolve('Failed to send question');
      }
    });
  }

  /**
   * Send issue selection/mode switch to the server
   */
  sendIssueSelection(mode: 'create' | 'update', issueNumber?: number, issueTitle?: string): boolean {
    console.log('[WebSocketService] sendIssueSelection called:', mode, issueNumber, issueTitle);
    
    if (!this.isConnected()) {
      console.warn('Cannot send issue selection: WebSocket not connected');
      return false;
    }

    try {
      const message: any = {
        type: 'issue_selection',
        mode,
        timestamp: Date.now(),
      };

      if (mode === 'update' && issueNumber) {
        message.issue_number = issueNumber;
        message.issue_title = issueTitle || '';
      }

      console.log('[WebSocketService] Sending issue selection message:', JSON.stringify(message));
      this.ws!.send(JSON.stringify(message));
      console.log('[WebSocketService] Issue selection sent successfully');
      return true;
    } catch (error) {
      console.error('Failed to send issue selection:', error);
      return false;
    }
  }

  /**
   * Send a combined frame and text message
   */
  sendFrameWithText(base64Image: string, text: string, width?: number, height?: number): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot send frame with text: WebSocket not connected');
      return false;
    }

    try {
      const message = {
        frameNumber: this.frameNumber++,
        timestamp: Date.now(),
        data: {
          base64Image,
          width,
          height,
          text,
        },
      };

      this.ws!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to send frame with text:', error);
      return false;
    }
  }

  /**
   * Request production tools from main branch
   */
  requestProductionTools(): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot request production tools: WebSocket not connected');
      return false;
    }

    try {
      const message = {
        type: 'request_production_tools',
        branch: Config.PRODUCTION_BRANCH,
        timestamp: Date.now(),
      };

      console.log('[WebSocketService] Requesting production tools from branch:', Config.PRODUCTION_BRANCH);
      this.ws!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to request production tools:', error);
      return false;
    }
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Set callback for connection state changes
   */
  onConnectionChange(callback: (connected: boolean) => void) {
    this.onConnectionChangeCallback = callback;
    // Immediately call with current connection state (for hot reload scenarios)
    if (callback) {
      callback(this.isConnected());
    }
  }

  /**
   * Set callback for server messages
   */
  onMessage(callback: (message: ServerMessage) => void) {
    this.onMessageCallback = callback;
  }

  /**
   * Set callback for errors
   */
  onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }

  /**
   * Reset frame counter
   */
  resetFrameCounter() {
    this.frameNumber = 0;
  }

  /**
   * Get heartbeat status for debugging
   */
  getHeartbeatStatus() {
    return {
      isHeartbeatActive: this.heartbeatInterval !== null,
      missedHeartbeats: this.missedHeartbeats,
      maxMissedHeartbeats: this.MAX_MISSED_HEARTBEATS,
      intervalMs: this.HEARTBEAT_INTERVAL_MS,
      timeoutMs: this.HEARTBEAT_TIMEOUT_MS,
    };
  }

  /**
   * Set the user ID for multi-user support
   */
  /**
   * Request issue list from server
   */
  requestIssueList(): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot request issue list: WebSocket not connected');
      return false;
    }

    try {
      const message = {
        type: 'request_issue_list',
        timestamp: Date.now(),
      };
      
      this.ws!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to request issue list:', error);
      return false;
    }
  }

  /**
   * Request PR list from server
   */
  requestPRList(): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot request PR list: WebSocket not connected');
      return false;
    }

    try {
      const message = {
        type: 'request_pr_list',
        timestamp: Date.now(),
      };
      
      this.ws!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to request PR list:', error);
      return false;
    }
  }

  /**
   * Request tools for a specific PR
   */
  requestPRTools(prNumber: number): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot request PR tools: WebSocket not connected');
      return false;
    }

    try {
      const message = {
        type: 'request_pr_tools',
        pr_number: prNumber,
        timestamp: Date.now(),
      };
      
      this.ws!.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Failed to request PR tools:', error);
      return false;
    }
  }

  /**
   * Request Copilot sessions for a specific PR
   */
  requestPRSessions(prNumber: number): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot request PR sessions: WebSocket not connected');
      return false;
    }

    try {
      const message = {
        type: 'get_pr_sessions',
        pr_number: prNumber,
        timestamp: Date.now(),
      };
      
      this.ws!.send(JSON.stringify(message));
      console.log(`Requesting sessions for PR #${prNumber}`);
      return true;
    } catch (error) {
      console.error('Failed to request PR sessions:', error);
      return false;
    }
  }

  /**
   * Request summaries for a specific Copilot session
   */
  requestSessionSummaries(sessionId: string): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot request session summaries: WebSocket not connected');
      return false;
    }

    try {
      const message = {
        type: 'get_session_summaries',
        session_id: sessionId,
        timestamp: Date.now(),
      };
      
      this.ws!.send(JSON.stringify(message));
      console.log(`Requesting summaries for session ${sessionId}`);
      return true;
    } catch (error) {
      console.error('Failed to request session summaries:', error);
      return false;
    }
  }

  /**
   * Request full logs for a specific Copilot session
   */
  requestSessionLogs(sessionId: string, startEntry?: number, endEntry?: number): boolean {
    if (!this.isConnected()) {
      console.warn('Cannot request session logs: WebSocket not connected');
      return false;
    }

    try {
      const message: any = {
        type: 'get_session_logs',
        session_id: sessionId,
        timestamp: Date.now(),
      };

      if (startEntry !== undefined) {
        message.start_entry = startEntry;
      }
      if (endEntry !== undefined) {
        message.end_entry = endEntry;
      }
      
      this.ws!.send(JSON.stringify(message));
      console.log(`Requesting logs for session ${sessionId}`);
      return true;
    } catch (error) {
      console.error('Failed to request session logs:', error);
      return false;
    }
  }
}

// Export singleton instance
export default new WebSocketService();
