import React from 'react';
import { LoadingSpinner } from './LoadingSpinner';

interface VerificationPanelProps {
  isLoading: boolean;
  error: string | null;
  result: string;
}

export const VerificationPanel: React.FC<VerificationPanelProps> = ({ isLoading, error, result }) => {
  return (
    <div className="bg-gray-800 rounded-lg shadow-xl border border-green-500">
      <h2 className="text-lg font-semibold p-3 bg-gray-700/50 rounded-t-lg border-b border-gray-600 text-green-400 flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Real Verification Result (Remote Execution)
      </h2>
      <div className="p-4 font-mono text-sm text-gray-300 min-h-[100px] max-h-96 overflow-y-auto">
        {isLoading && (
          <div className="flex items-center justify-center gap-3 text-gray-400 h-full">
            <LoadingSpinner />
            <span>Executing on remote server (134.199.201.182)...</span>
          </div>
        )}
        {error && <div className="text-red-400 bg-red-900/50 p-3 rounded-md">{error}</div>}
        {result && !isLoading && (
          <pre className="whitespace-pre-wrap break-words">{result}</pre>
        )}
      </div>
      <div className="p-2 text-xs text-center text-gray-400 bg-gray-900/50 rounded-b-lg border-t border-gray-700">
        ✅ Real execution on remote GPU server via SSH | Environment: Triton-Puzzles/triton_env
      </div>
    </div>
  );
};
