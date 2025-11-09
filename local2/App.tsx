import React, { useState, useCallback } from 'react';
import { Language } from './types';
import { LANGUAGES, DEFAULT_CUDA_CODE } from './constants';
import { translateCode, verifyCode, testSSHConnection } from './services/geminiService';
import { CodePanel } from './components/CodePanel';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ArrowIcon } from './components/ArrowIcon';
import { VerificationPanel } from './components/VerificationPanel';
import { ActivityLog, ActivityEntry } from './components/ActivityLog';

const App: React.FC = () => {
  const [sourceLanguage, setSourceLanguage] = useState<Language>(Language.CUDA);
  const [targetLanguage, setTargetLanguage] = useState<Language>(Language.TRITON);
  const [sourceCode, setSourceCode] = useState<string>(DEFAULT_CUDA_CODE);
  const [translatedCode, setTranslatedCode] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [verificationResult, setVerificationResult] = useState<string>('');
  const [isVerifying, setIsVerifying] = useState<boolean>(false);
  const [verificationError, setVerificationError] = useState<string | null>(null);

  const [feedback, setFeedback] = useState<string>('');
  const [showFeedback, setShowFeedback] = useState<boolean>(false);
  
  // Track previous verification attempt for feedback
  const [previousPackagedScript, setPreviousPackagedScript] = useState<string>('');
  const [previousVerificationError, setPreviousVerificationError] = useState<string>('');

  const [isTestingSSH, setIsTestingSSH] = useState<boolean>(false);
  const [sshTestResult, setSSHTestResult] = useState<string>('');

  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [showActivityLog, setShowActivityLog] = useState<boolean>(true);

  const addActivity = useCallback((
    type: ActivityEntry['type'],
    action: string,
    message: string
  ) => {
    const newActivity: ActivityEntry = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      type,
      action,
      message,
    };
    setActivities(prev => [...prev, newActivity]);
  }, []);

  const clearActivities = useCallback(() => {
    setActivities([]);
  }, []);

  const handleTranslate = useCallback(async () => {
    if (!sourceCode.trim()) {
      setError('Source code cannot be empty.');
      addActivity('error', 'Translation Failed', 'Source code is empty');
      return;
    }
    if (sourceLanguage === targetLanguage) {
        setError('Source and target languages cannot be the same.');
        addActivity('error', 'Translation Failed', 'Source and target languages must be different');
        return;
    }

    addActivity('info', 'Translate Button Pressed', `Translating from ${sourceLanguage} to ${targetLanguage}`);

    setIsLoading(true);
    setError(null);
    setTranslatedCode('');
    setVerificationResult('');
    setVerificationError(null);
    setShowFeedback(false);
    setFeedback('');

    try {
      addActivity('info', 'Calling Ollama', `Model: gemma3n:latest | Task: Code translation`);
      const result = await translateCode(sourceLanguage, targetLanguage, sourceCode);
      setTranslatedCode(result);
      addActivity('success', 'Translation Complete', `Successfully translated ${sourceCode.split('\n').length} lines of ${sourceLanguage} code to ${targetLanguage}`);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(errorMsg);
      addActivity('error', 'Translation Failed', errorMsg);
    } finally {
      setIsLoading(false);
    }
  }, [sourceLanguage, targetLanguage, sourceCode, addActivity]);

  const handleRegenerate = useCallback(async () => {
    if (!feedback.trim()) {
      setError('Feedback cannot be empty.');
      addActivity('error', 'Regeneration Failed', 'Feedback is required');
      return;
    }

    addActivity('info', 'Regenerate Button Pressed', 'Regenerating code with user feedback');

    setIsLoading(true);
    setError(null);
    setVerificationResult('');
    setVerificationError(null);

    try {
      addActivity('info', 'Calling Ollama', `Model: gemma3n:latest | Task: Code regeneration with feedback`);
      const result = await translateCode(sourceLanguage, targetLanguage, sourceCode, {
        previousCode: translatedCode,
        userFeedback: feedback,
      });
      setTranslatedCode(result);
      setShowFeedback(false);
      setFeedback('');
      addActivity('success', 'Regeneration Complete', 'Code regenerated based on feedback');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred during regeneration.';
      setError(errorMsg);
      addActivity('error', 'Regeneration Failed', errorMsg);
    } finally {
      setIsLoading(false);
    }
  }, [sourceLanguage, targetLanguage, sourceCode, translatedCode, feedback, addActivity]);

  const handleVerify = useCallback(async () => {
    if (!translatedCode.trim()) {
      setVerificationError('There is no translated code to verify.');
      addActivity('error', 'Verification Failed', 'No translated code available');
      return;
    }

    // Check if we should use feedback for re-verification
    // Use feedback if: (1) user provided feedback, OR (2) we have previous error from failed verification
    const hasFeedback = feedback.trim() !== '';
    const hasPreviousError = previousPackagedScript !== '' && previousVerificationError.trim() !== '';
    const isReVerification = hasFeedback || hasPreviousError;
    
    if (isReVerification) {
      addActivity('info', 'Verify Kernel Button Pressed (Re-verification)', `🔄 Re-verifying with feedback${hasFeedback ? ' (user feedback provided)' : ' (using previous error)'}`);
    } else {
      addActivity('info', 'Verify Kernel Button Pressed', `Verifying ${targetLanguage} code on remote server`);
    }

    setIsVerifying(true);
    setVerificationError(null);
    setVerificationResult('');

    try {
      if (isReVerification) {
        addActivity('info', 'Step 1: Re-Packaging Code', '🔄 Calling Ollama with feedback to create corrected Python script...');
      } else {
        addActivity('info', 'Step 1: Packaging Code', 'Calling Ollama to create executable Python script...');
      }
      
      // Prepare feedback if available
      const verificationFeedback = isReVerification ? {
        previousScript: previousPackagedScript,
        userFeedback: hasFeedback ? feedback : 'The previous script failed. Please fix all errors and generate a working version.',
        previousError: previousVerificationError
      } : undefined;
      
      const { result, packagedScript, executionError } = await verifyCode(
        targetLanguage, 
        translatedCode,
        verificationFeedback
      );
      
      // Save for potential re-verification
      setPreviousPackagedScript(packagedScript);
      setPreviousVerificationError(executionError);
      
      // Clear user feedback after using it (but keep tracking previous errors for auto-retry)
      if (hasFeedback) {
        setFeedback('');
        setShowFeedback(false);
      }
      
      // Extract debug file path from logs if present
      const debugFileMatch = result.match(/Debug: Script saved to (.+\.py)/);
      if (debugFileMatch) {
        addActivity('info', 'Debug File Saved', `📝 Packaged script saved to: ${debugFileMatch[1]}`);
      }
      
      addActivity('info', 'Step 2: SSH Connection', 'Using: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182');
      addActivity('info', 'Step 3: Code Upload', 'Uploading script to remote server');
      addActivity('info', 'Step 4: Execution', 'Activating Triton environment and running code');
      addActivity('info', 'Step 5: Results', 'Capturing stdout, stderr, and exit code');
      addActivity('info', 'Step 6: Cleanup', 'Removing temporary files');
      
      setVerificationResult(result);
      
      if (result.includes('COMPILATION STATUS: SUCCESS')) {
        addActivity('success', 'Verification Complete', '✅ Code executed successfully on remote server');
      } else {
        addActivity('warning', 'Verification Complete', '⚠️ Execution completed with errors - check logs and provide feedback');
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred during verification.';
      setVerificationError(errorMsg);
      addActivity('error', 'Verification Failed', errorMsg);
    } finally {
      setIsVerifying(false);
    }
  }, [targetLanguage, translatedCode, feedback, previousPackagedScript, previousVerificationError, addActivity]);

  const handleTestSSH = useCallback(async () => {
    addActivity('info', 'Test SSH Button Pressed', 'Testing connection to remote server');

    setIsTestingSSH(true);
    setSSHTestResult('');
    setVerificationError(null);

    try {
      addActivity('info', 'SSH Connection Test', 'Connecting to root@134.199.201.182...');
      const result = await testSSHConnection();
      if (result.success) {
        setSSHTestResult(result.logs);
        addActivity('success', 'SSH Test Complete', '✅ Connection successful, ls command executed');
      } else {
        setVerificationError(result.error || result.message);
        setSSHTestResult(result.logs);
        addActivity('error', 'SSH Test Failed', result.error || result.message);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'An unknown error occurred during SSH test.';
      setVerificationError(errorMsg);
      addActivity('error', 'SSH Test Failed', errorMsg);
    } finally {
      setIsTestingSSH(false);
    }
  }, [addActivity]);

  const swapLanguages = () => {
    setSourceLanguage(targetLanguage);
    setTargetLanguage(sourceLanguage);
    setSourceCode(translatedCode);
    setTranslatedCode(sourceCode);
    setVerificationResult('');
    setVerificationError(null);
    setShowFeedback(false);
    setFeedback('');
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 font-sans flex flex-col">
      <header className="py-4 px-6 shadow-lg bg-gray-800/50 backdrop-blur-sm border-b border-gray-700">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex-1"></div>
          <div className="text-center">
            <h1 className="text-2xl font-bold text-cyan-400">GPU Kernel Translator</h1>
            <p className="text-gray-400 text-sm">Translate CUDA, Triton, and Mojo Kernels with Ollama | Verify on Remote Server</p>
          </div>
          <div className="flex-1 flex justify-end">
            <button
              onClick={handleTestSSH}
              disabled={isTestingSSH}
              className="px-4 py-2 bg-purple-600 text-white text-sm font-semibold rounded-md hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              title="Test SSH connection to remote server"
            >
              {isTestingSSH ? (
                <>
                  <LoadingSpinner />
                  <span>Testing...</span>
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Test SSH</span>
                </>
              )}
            </button>
          </div>
        </div>
      </header>
      
      <main className="flex-grow flex gap-4 p-4 lg:p-6 overflow-hidden min-h-0">
        {/* Main Content Area - Code Panels */}
        <div className="flex-1 flex flex-col gap-4 min-w-0 min-h-0 overflow-auto">
          <div className="flex-1 flex flex-col lg:flex-row gap-4 min-h-0">
            <CodePanel
              label="Source"
              language={sourceLanguage}
              onLanguageChange={setSourceLanguage}
              code={sourceCode}
              onCodeChange={setSourceCode}
            />
            <div className="flex justify-center items-center">
              <button
                onClick={swapLanguages}
                className="p-2 bg-gray-700 rounded-full hover:bg-cyan-500 transition-colors duration-200 transform lg:rotate-0 rotate-90"
                title="Swap Languages"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
              </button>
            </div>
            <CodePanel
              label="Translation"
              language={targetLanguage}
              onLanguageChange={setTargetLanguage}
              code={translatedCode}
              isReadOnly
              onVerify={handleVerify}
              isVerifying={isVerifying}
              onShowFeedback={() => setShowFeedback(!showFeedback)}
            />
          </div>
        
        {showFeedback && (
          <div className="bg-gray-800 rounded-lg shadow-xl border border-amber-500 p-4 flex flex-col gap-3">
            <h3 className="text-md font-semibold text-amber-400">Provide Feedback</h3>
            <p className="text-sm text-gray-400">
              Paste any errors or describe issues below. You can then either:
              <br />• Click <strong>"Regenerate Translation"</strong> to fix the translated code
              <br />• Click <strong>"Verify Kernel"</strong> to re-package with corrections
            </p>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="e.g., ModuleNotFoundError: No module named 'triton'&#10;or&#10;SyntaxError: invalid syntax at line 15..."
              className="w-full h-32 p-3 bg-gray-900 text-gray-300 font-mono text-sm resize-y rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-amber-500"
              spellCheck="false"
            />
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowFeedback(false)}
                className="px-4 py-2 bg-gray-600 text-white font-semibold text-sm rounded-md hover:bg-gray-500 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleRegenerate}
                disabled={isLoading}
                className="flex items-center justify-center gap-2 px-4 py-2 bg-amber-600 text-white font-semibold text-sm rounded-md hover:bg-amber-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? (
                  <>
                    <LoadingSpinner />
                    <span>Regenerating...</span>
                  </>
                ) : (
                  'Regenerate Translation'
                )}
              </button>
            </div>
          </div>
        )}

        {(isTestingSSH || sshTestResult) && (
            <div className="bg-gray-800 rounded-lg shadow-xl border border-purple-500 p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h3 className="text-md font-semibold text-purple-400 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  SSH Connection Test
                </h3>
                <button
                  onClick={() => setSSHTestResult('')}
                  className="text-gray-400 hover:text-white transition-colors"
                  title="Close"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              {isTestingSSH ? (
                <div className="flex items-center justify-center gap-3 py-8">
                  <LoadingSpinner />
                  <span className="text-gray-400">Connecting to remote server...</span>
                </div>
              ) : sshTestResult ? (
                <div className="bg-gray-900 rounded-md p-4 font-mono text-sm text-gray-300 whitespace-pre-wrap overflow-x-auto max-h-96 overflow-y-auto border border-gray-700">
                  {sshTestResult}
                </div>
              ) : null}
            </div>
        )}

          {/* Auto-retry indicator */}
          {!isVerifying && previousPackagedScript && previousVerificationError && !showFeedback && (
            <div className="bg-blue-900/30 border border-blue-500/50 rounded-lg p-3 flex items-start gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div className="flex-1">
                <p className="text-sm text-blue-300 font-semibold mb-1">🔄 Auto-Retry Available</p>
                <p className="text-xs text-blue-200/80">
                  Previous verification failed. Click <strong>"Verify Kernel"</strong> again to automatically retry with error corrections.
                  Or click <strong>"Provide Feedback"</strong> to add specific instructions.
                </p>
              </div>
              <button
                onClick={() => {
                  setPreviousPackagedScript('');
                  setPreviousVerificationError('');
                }}
                className="text-blue-400 hover:text-blue-300 transition-colors"
                title="Clear auto-retry"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          {(isVerifying || verificationResult || verificationError) && !sshTestResult && (
            <VerificationPanel
              isLoading={isVerifying}
              error={verificationError}
              result={verificationResult}
            />
          )}
        </div>
        
        {/* Activity Log Sidebar */}
        <div className={`flex flex-col transition-all duration-300 flex-shrink-0 ${showActivityLog ? 'w-80' : 'w-12'}`}>
          {/* Toggle Button */}
          <button
            onClick={() => setShowActivityLog(!showActivityLog)}
            className="flex items-center justify-center p-2 bg-gray-800 hover:bg-gray-700 rounded-lg mb-2 transition-colors border border-gray-700 flex-shrink-0"
            title={showActivityLog ? 'Hide Activity Log' : 'Show Activity Log'}
          >
            {showActivityLog ? (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
                <span className="ml-2 text-sm text-gray-400">Hide Log</span>
              </>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            )}
          </button>
          
          {/* Activity Log */}
          {showActivityLog && (
            <div className="flex-1 min-h-0 overflow-hidden">
              <ActivityLog activities={activities} onClear={clearActivities} />
            </div>
          )}
        </div>
      </main>

      <footer className="sticky bottom-0 left-0 right-0 p-4 bg-gray-800/50 backdrop-blur-sm border-t border-gray-700">
        <div className="max-w-4xl mx-auto flex flex-col items-center gap-4">
          {error && <div className="text-red-400 bg-red-900/50 p-3 rounded-md w-full text-center">{error}</div>}
          <button
            onClick={handleTranslate}
            disabled={isLoading}
            className="w-full max-w-xs flex items-center justify-center gap-3 px-8 py-4 bg-cyan-600 text-white font-bold rounded-lg shadow-lg hover:bg-cyan-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
          >
            {isLoading && !showFeedback ? (
              <>
                <LoadingSpinner />
                <span>Translating...</span>
              </>
            ) : (
              <>
                <span>Translate</span>
                <ArrowIcon />
              </>
            )}
          </button>
        </div>
      </footer>
    </div>
  );
};

export default App;