import React from 'react';
import { Language } from '../types';
import { LANGUAGES } from '../constants';
import { LoadingSpinner } from './LoadingSpinner';

interface CodePanelProps {
  label: string;
  language: Language;
  onLanguageChange: (language: Language) => void;
  code: string;
  onCodeChange?: (code: string) => void;
  isReadOnly?: boolean;
  onVerify?: () => void;
  isVerifying?: boolean;
  onShowFeedback?: () => void;
}

export const CodePanel: React.FC<CodePanelProps> = ({
  label,
  language,
  onLanguageChange,
  code,
  onCodeChange,
  isReadOnly = false,
  onVerify,
  isVerifying = false,
  onShowFeedback,
}) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="flex flex-col flex-1 bg-gray-800 rounded-lg shadow-xl border border-gray-700 min-h-[300px] lg:min-h-0">
      <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded-t-lg border-b border-gray-600">
        <div className="flex items-center gap-4">
          <label htmlFor={`${label}-lang`} className="text-sm font-semibold text-gray-400">
            {label}
          </label>
          <select
            id={`${label}-lang`}
            value={language}
            onChange={(e) => onLanguageChange(e.target.value as Language)}
            className="bg-gray-800 border border-gray-600 rounded-md px-2 py-1 text-sm text-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            {LANGUAGES.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
        </div>
        {isReadOnly && code && (
          <button
            onClick={handleCopy}
            className="text-sm bg-gray-600 hover:bg-gray-500 px-3 py-1 rounded-md transition-colors"
          >
            Copy
          </button>
        )}
      </div>
      <div className="relative flex-1">
        <textarea
          value={code}
          onChange={(e) => onCodeChange && onCodeChange(e.target.value)}
          readOnly={isReadOnly}
          placeholder={isReadOnly ? 'Translation will appear here...' : 'Paste your kernel code here...'}
          className="w-full h-full p-4 bg-transparent text-gray-300 font-mono text-sm resize-none focus:outline-none absolute top-0 left-0"
          spellCheck="false"
        />
      </div>
      {isReadOnly && code && (onVerify || onShowFeedback) && (
        <div className="p-3 bg-gray-700/50 rounded-b-lg border-t border-gray-600 flex justify-end items-center gap-3">
          {onShowFeedback && (
             <button
              onClick={onShowFeedback}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-amber-600 text-white font-semibold text-sm rounded-md shadow-md hover:bg-amber-500 transition-all duration-200"
            >
              <span>Report Error &amp; Regenerate</span>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </button>
          )}
          {onVerify && (
            <button
              onClick={onVerify}
              disabled={isVerifying}
              className="flex items-center justify-center gap-2 px-4 py-2 bg-indigo-600 text-white font-semibold text-sm rounded-md shadow-md hover:bg-indigo-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-all duration-200"
            >
              {isVerifying ? (
                <>
                  <LoadingSpinner />
                  <span>Verifying...</span>
                </>
              ) : (
                <>
                  <span>Verify Kernel</span>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </>
              )}
            </button>
          )}
        </div>
      )}
    </div>
  );
};