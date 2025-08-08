// TPS Mobile App - React Native Interface
// Advanced reasoning companion for iOS and Android

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  Animated,
  Dimensions,
  StatusBar,
  SafeAreaView,
  ActivityIndicator,
  Modal,
  Share,
  Haptics,
  Platform
} from 'react-native';

import AsyncStorage from '@react-native-async-storage/async-storage';
import { LineChart, RadarChart } from 'react-native-chart-kit';
import PushNotification from 'react-native-push-notification';
import Voice from '@react-native-voice/voice';
import * as Keychain from 'react-native-keychain';
import NetInfo from '@react-native-community/netinfo';
import Orientation from 'react-native-orientation-locker';

// Custom Components
import { TPSVisualization } from './components/TPSVisualization';
import { WaveProgressIndicator } from './components/WaveProgressIndicator';
import { InsightCard } from './components/InsightCard';
import { SessionHistory } from './components/SessionHistory';
import { ConfigurationPicker } from './components/ConfigurationPicker';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// Main TPS Mobile App Component
export default function TPSMobileApp() {
  // State Management
  const [userInput, setUserInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentSession, setCurrentSession] = useState(null);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userProfile, setUserProfile] = useState(null);
  const [selectedConfiguration, setSelectedConfiguration] = useState('default');
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [networkStatus, setNetworkStatus] = useState('connected');
  const [offlineQueue, setOfflineQueue] = useState([]);

  // Animations
  const fadeAnim = new Animated.Value(0);
  const slideAnim = new Animated.Value(screenHeight);
  const pulseAnim = new Animated.Value(1);

  // Configuration
  const API_BASE_URL = __DEV__ 
    ? 'http://localhost:8000' 
    : 'https://api.tps-reasoning.com';

  // Initialize App
  useEffect(() => {
    initializeApp();
    setupNetworkMonitoring();
    setupVoiceRecognition();
    setupNotifications();
    
    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, []);

  const initializeApp = async () => {
    try {
      // Check authentication
      const credentials = await Keychain.getInternetCredentials('tps-app');
      if (credentials) {
        setIsAuthenticated(true);
        await loadUserProfile();
        await loadSessionHistory();
      }

      // Animate app entrance
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.spring(slideAnim, {
          toValue: 0,
          tension: 50,
          friction: 8,
          useNativeDriver: true,
        })
      ]).start();

    } catch (error) {
      console.error('App initialization error:', error);
    }
  };

  const setupNetworkMonitoring = () => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setNetworkStatus(state.isConnected ? 'connected' : 'disconnected');
      
      // Process offline queue when connected
      if (state.isConnected && offlineQueue.length > 0) {
        processOfflineQueue();
      }
    });

    return unsubscribe;
  };

  const setupVoiceRecognition = () => {
    Voice.onSpeechStart = () => {
      setIsListening(true);
      startPulseAnimation();
    };

    Voice.onSpeechEnd = () => {
      setIsListening(false);
      stopPulseAnimation();
    };

    Voice.onSpeechResults = (e) => {
      if (e.value && e.value.length > 0) {
        setUserInput(e.value[0]);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      }
    };

    Voice.onSpeechError = (e) => {
      console.error('Voice recognition error:', e.error);
      setIsListening(false);
      stopPulseAnimation();
    };
  };

  const setupNotifications = () => {
    PushNotification.configure({
      onNotification: function(notification) {
        console.log('Notification:', notification);
      },
      permissions: {
        alert: true,
        badge: true,
        sound: true,
      },
      popInitialNotification: true,
      requestPermissions: Platform.OS === 'ios',
    });
  };

  // Authentication Functions
  const handleLogin = async (credentials) => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (response.ok) {
        await Keychain.setInternetCredentials(
          'tps-app',
          credentials.username,
          data.token
        );
        setIsAuthenticated(true);
        await loadUserProfile();
        return true;
      } else {
        Alert.alert('Login Error', data.error || 'Authentication failed');
        return false;
      }
    } catch (error) {
      console.error('Login error:', error);
      Alert.alert('Network Error', 'Please check your connection');
      return false;
    }
  };

  const handleLogout = async () => {
    try {
      await Keychain.resetInternetCredentials('tps-app');
      setIsAuthenticated(false);
      setUserProfile(null);
      setSessionHistory([]);
      setCurrentSession(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // Core TPS Processing
  const processReasoning = async (inputText = userInput) => {
    if (!inputText.trim()) {
      Alert.alert('Input Required', 'Please enter your situation or question');
      return;
    }

    setIsProcessing(true);
    
    try {
      // Haptic feedback
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      const credentials = await Keychain.getInternetCredentials('tps-app');
      
      // Check network status
      if (networkStatus === 'disconnected') {
        await handleOfflineProcessing(inputText);
        return;
      }

      const response = await fetch(`${API_BASE_URL}/reason`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${credentials.password}`,
        },
        body: JSON.stringify({
          input: inputText,
          configuration: selectedConfiguration,
        }),
      });

      const sessionData = await response.json();

      if (response.ok) {
        setCurrentSession(sessionData);
        await saveSessionToHistory(sessionData);
        setUserInput('');
        
        // Success haptic
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        
        // Schedule reflection notification
        scheduleReflectionNotification(sessionData);
        
      } else {
        throw new Error(sessionData.error || 'Processing failed');
      }

    } catch (error) {
      console.error('Processing error:', error);
      Alert.alert('Processing Error', error.message);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleOfflineProcessing = async (inputText) => {
    // Basic offline TPS analysis using cached patterns
    const offlineSession = {
      session_id: `offline_${Date.now()}`,
      input_text: inputText,
      timestamp: new Date().toISOString(),
      offline_mode: true,
      tps_analysis: {
        scores: 'E5.0L5.0H5.0I5.0', // Default balanced scores
        note: 'Processed offline - connect to internet for full analysis'
      },
      insights: ['This session was processed offline. Connect to get full TPS analysis.'],
      success_metrics: { offline_processing: 1.0 }
    };

    setCurrentSession(offlineSession);
    
    // Add to offline queue for later sync
    setOfflineQueue(prev => [...prev, { input: inputText, timestamp: Date.now() }]);
    
    Alert.alert(
      'Offline Mode', 
      'Limited processing available offline. Full analysis will sync when connected.'
    );
  };

  const processOfflineQueue = async () => {
    try {
      const credentials = await Keychain.getInternetCredentials('tps-app');
      
      for (const queuedItem of offlineQueue) {
        await fetch(`${API_BASE_URL}/reason`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${credentials.password}`,
          },
          body: JSON.stringify({
            input: queuedItem.input,
            configuration: selectedConfiguration,
            offline_sync: true,
          }),
        });
      }

      setOfflineQueue([]);
      await loadSessionHistory(); // Refresh with synced sessions
      
    } catch (error) {
      console.error('Offline sync error:', error);
    }
  };

  // Voice Processing
  const startVoiceRecognition = async () => {
    try {
      await Voice.start('en-US');
      setIsVoiceMode(true);
    } catch (error) {
      console.error('Voice start error:', error);
      Alert.alert('Voice Error', 'Could not start voice recognition');
    }
  };

  const stopVoiceRecognition = async () => {
    try {
      await Voice.stop();
      setIsVoiceMode(false);
    } catch (error) {
      console.error('Voice stop error:', error);
    }
  };

  // Animation Functions
  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const stopPulseAnimation = () => {
    pulseAnim.setValue(1);
  };

  // Utility Functions
  const saveSessionToHistory = async (session) => {
    try {
      const history = await AsyncStorage.getItem('tps_session_history');
      const sessions = history ? JSON.parse(history) : [];
      sessions.unshift({ ...session, timestamp: new Date().toISOString() });
      
      // Keep only last 50 sessions
      const recentSessions = sessions.slice(0, 50);
      
      await AsyncStorage.setItem('tps_session_history', JSON.stringify(recentSessions));
      setSessionHistory(recentSessions);
    } catch (error) {
      console.error('Error saving session:', error);
    }
  };

  const loadSessionHistory = async () => {
    try {
      const history = await AsyncStorage.getItem('tps_session_history');
      if (history) {
        setSessionHistory(JSON.parse(history));
      }
    } catch (error) {
      console.error('Error loading session history:', error);
    }
  };

  const loadUserProfile = async () => {
    try {
      const credentials = await Keychain.getInternetCredentials('tps-app');
      const response = await fetch(`${API_BASE_URL}/user/profile`, {
        headers: {
          'Authorization': `Bearer ${credentials.password}`,
        },
      });

      if (response.ok) {
        const profile = await response.json();
        setUserProfile(profile);
      }
    } catch (error) {
      console.error('Error loading user profile:', error);
    }
  };

  const shareSession = async (session) => {
    try {
      const shareContent = `🧠 TPS Analysis Results\n\nInput: ${session.input_text}\n\nTPS Scores: ${session.tps_analysis?.scores}\n\nKey Insights:\n${session.insights?.map(insight => `• ${insight}`).join('\n')}\n\nGenerated by TPS Reasoning Engine v6`;
      
      await Share.share({
        message: shareContent,
        title: 'TPS Reasoning Session',
      });
    } catch (error) {
      console.error('Error sharing session:', error);
    }
  };

  const scheduleReflectionNotification = (session) => {
    // Schedule notification for 24 hours later
    const reflectionTime = new Date();
    reflectionTime.setHours(reflectionTime.getHours() + 24);

    PushNotification.localNotificationSchedule({
      message: `How are you feeling about yesterday's insight: "${session.insights?.[0]?.substring(0, 50)}..."?`,
      date: reflectionTime,
      playSound: true,
      soundName: 'default',
    });
  };

  // Render Authentication Screen
  const renderAuthScreen = () => (
    <SafeAreaView style={styles.authContainer}>
      <Animated.View style={[styles.authContent, { opacity: fadeAnim }]}>
        <Text style={styles.authTitle}>🌊 TPS Reasoning</Text>
        <Text style={styles.authSubtitle}>Advanced Wave Intelligence</Text>
        
        <LoginForm onLogin={handleLogin} />
      </Animated.View>
    </SafeAreaView>
  );

  // Render Main App Interface
  const renderMainApp = () => (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#667eea" />
      
      <Animated.View style={[
        styles.content,
        {
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }]
        }
      ]}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>TPS Reasoning v6</Text>
          <TouchableOpacity
            style={styles.profileButton}
            onPress={() => setShowProfile(true)}
          >
            <Text style={styles.profileText}>
              {userProfile?.username?.[0]?.toUpperCase() || 'U'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Network Status Indicator */}
        {networkStatus === 'disconnected' && (
          <View style={styles.networkWarning}>
            <Text style={styles.networkWarningText}>
              📡 Offline Mode - Limited functionality
            </Text>
          </View>
        )}

        <ScrollView style={styles.scrollContent} showsVerticalScrollIndicator={false}>
          {/* Configuration Selector */}
          <ConfigurationPicker
            selectedConfig={selectedConfiguration}
            onConfigChange={setSelectedConfiguration}
          />

          {/* Input Section */}
          <View style={styles.inputSection}>
            <Text style={styles.inputLabel}>What's on your mind?</Text>
            
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.textInput}
                multiline
                placeholder="Share your situation, thoughts, or questions here..."
                placeholderTextColor="#a0a0a0"
                value={userInput}
                onChangeText={setUserInput}
                maxLength={5000}
              />
              
              <TouchableOpacity
                style={[styles.voiceButton, isListening && styles.voiceButtonActive]}
                onPress={isListening ? stopVoiceRecognition : startVoiceRecognition}
              >
                <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
                  <Text style={styles.voiceButtonText}>
                    {isListening ? '🔴' : '🎤'}
                  </Text>
                </Animated.View>
              </TouchableOpacity>
            </View>

            <TouchableOpacity
              style={[styles.processButton, isProcessing && styles.processButtonDisabled]}
              onPress={processReasoning}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={styles.processButtonText}>🌊 Begin Wave Reasoning</Text>
              )}
            </TouchableOpacity>
          </View>

          {/* Current Session Results */}
          {currentSession && (
            <Animated.View style={styles.resultsSection}>
              <View style={styles.resultHeader}>
                <Text style={styles.resultTitle}>TPS Analysis Results</Text>
                <TouchableOpacity
                  style={styles.shareButton}
                  onPress={() => shareSession(currentSession)}
                >
                  <Text style={styles.shareButtonText}>Share</Text>
                </TouchableOpacity>
              </View>

              {/* TPS Visualization */}
              <TPSVisualization session={currentSession} />

              {/* Wave Progress */}
              <WaveProgressIndicator session={currentSession} />

              {/* Insights */}
              {currentSession.insights?.map((insight, index) => (
                <InsightCard key={index} insight={insight} />
              ))}

              {/* Success Metrics */}
              {currentSession.success_metrics && (
                <View style={styles.metricsContainer}>
                  <Text style={styles.metricsTitle}>Success Score</Text>
                  <Text style={styles.metricsValue}>
                    {(currentSession.success_metrics.overall_success * 100).toFixed(0)}%
                  </Text>
                </View>
              )}
            </Animated.View>
          )}

          {/* Session History */}
          <SessionHistory
            sessions={sessionHistory}
            onSessionSelect={setCurrentSession}
          />
        </ScrollView>
      </Animated.View>
    </SafeAreaView>
  );

  // Main Render
  return isAuthenticated ? renderMainApp() : renderAuthScreen();
}

// Login Form Component
const LoginForm = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!username.trim() || !password.trim()) {
      Alert.alert('Error', 'Please enter username and password');
      return;
    }

    setLoading(true);
    const success = await onLogin({ username, password });
    setLoading(false);

    if (success) {
      setUsername('');
      setPassword('');
    }
  };

  return (
    <View style={styles.loginForm}>
      <TextInput
        style={styles.loginInput}
        placeholder="Username"
        placeholderTextColor="#a0a0a0"
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
        autoCorrect={false}
      />
      
      <TextInput
        style={styles.loginInput}
        placeholder="Password"
        placeholderTextColor="#a0a0a0"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        autoCapitalize="none"
        autoCorrect={false}
      />

      <TouchableOpacity
        style={[styles.loginButton, loading && styles.loginButtonDisabled]}
        onPress={handleSubmit}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator color="white" />
        ) : (
          <Text style={styles.loginButtonText}>Sign In</Text>
        )}
      </TouchableOpacity>
    </View>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  authContainer: {
    flex: 1,
    backgroundColor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  authContent: {
    width: screenWidth * 0.85,
    padding: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
  },
  authTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#667eea',
    marginBottom: 10,
  },
  authSubtitle: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginBottom: 30,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#667eea',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  profileButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  networkWarning: {
    backgroundColor: '#ff9500',
    padding: 10,
    alignItems: 'center',
  },
  networkWarningText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  scrollContent: {
    flex: 1,
  },
  inputSection: {
    padding: 20,
    backgroundColor: 'white',
    margin: 20,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  inputLabel: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 15,
  },
  inputContainer: {
    position: 'relative',
  },
  textInput: {
    borderWidth: 2,
    borderColor: '#e0e0e0',
    borderRadius: 12,
    padding: 15,
    fontSize: 16,
    minHeight: 120,
    textAlignVertical: 'top',
    paddingRight: 60, // Make room for voice button
  },
  voiceButton: {
    position: 'absolute',
    right: 10,
    bottom: 10,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#667eea',
    justifyContent: 'center',
    alignItems: 'center',
  },
  voiceButtonActive: {
    backgroundColor: '#ff4757',
  },
  voiceButtonText: {
    fontSize: 18,
  },
  processButton: {
    backgroundColor: '#667eea',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 15,
  },
  processButtonDisabled: {
    backgroundColor: '#ccc',
  },
  processButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  resultsSection: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  shareButton: {
    backgroundColor: '#28a745',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 8,
  },
  shareButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    marginTop: 15,
  },
  metricsTitle: {
    fontSize: 16,
    color: '#666',
  },
  metricsValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#667eea',
  },
  loginForm: {
    width: '100%',
  },
  loginInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 12,
    padding: 15,
    fontSize: 16,
    marginBottom: 15,
    backgroundColor: 'white',
  },
  loginButton: {
    backgroundColor: '#667eea',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 10,
  },
  loginButtonDisabled: {
    backgroundColor: '#ccc',
  },
  loginButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

// Export additional components that would be in separate files
export { TPSVisualization, WaveProgressIndicator, InsightCard, SessionHistory, ConfigurationPicker };
