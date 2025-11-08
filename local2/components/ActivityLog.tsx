import React, { useEffect, useRef } from 'react';

export interface ActivityEntry {
  id: string;
  timestamp: Date;
  type: 'info' | 'success' | 'error' | 'warning';
  action: string;
  message: string;
}

interface ActivityLogProps {
  activities: ActivityEntry[];
  onClear: () => void;
}

export const ActivityLog: React.FC<ActivityLogProps> = ({ activities, onClear }) => {
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new activities are added
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activities]);

  const getIcon = (type: ActivityEntry['type']) => {
    switch (type) {
      case 'success':
        return (
          <svg className="h-4 w-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'error':
        return (
          <svg className="h-4 w-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="h-4 w-4 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        );
      default:
        return (
          <svg className="h-4 w-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  const getTextColor = (type: ActivityEntry['type']) => {
    switch (type) {
      case 'success':
        return 'text-green-300';
      case 'error':
        return 'text-red-300';
      case 'warning':
        return 'text-yellow-300';
      default:
        return 'text-gray-300';
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      hour12: false 
    });
  };

  return (
    <div className="bg-gray-800 rounded-lg shadow-xl border border-blue-500">
      <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded-t-lg border-b border-gray-600">
        <h2 className="text-lg font-semibold text-blue-400 flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Activity Log
          {activities.length > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-blue-500/30 text-blue-300 text-xs rounded-full">
              {activities.length}
            </span>
          )}
        </h2>
        {activities.length > 0 && (
          <button
            onClick={onClear}
            className="px-3 py-1 text-xs bg-gray-600 hover:bg-gray-500 text-gray-300 rounded transition-colors"
          >
            Clear
          </button>
        )}
      </div>
      <div className="p-4 font-mono text-xs text-gray-300 max-h-80 overflow-y-auto bg-gray-900/50">
        {activities.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-gray-500">
            <div className="text-center">
              <svg className="mx-auto h-8 w-8 mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p>No activity yet</p>
              <p className="text-xs mt-1">Actions will appear here as you use the app</p>
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            {activities.map((activity) => (
              <div
                key={activity.id}
                className="flex items-start gap-3 p-2 rounded bg-gray-800/50 hover:bg-gray-800 transition-colors"
              >
                <div className="flex-shrink-0 mt-0.5">{getIcon(activity.type)}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-gray-500 text-[10px] font-mono">
                      {formatTime(activity.timestamp)}
                    </span>
                    <span className={`font-semibold ${getTextColor(activity.type)}`}>
                      {activity.action}
                    </span>
                  </div>
                  <p className="text-gray-400 text-xs break-words">{activity.message}</p>
                </div>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        )}
      </div>
      <div className="p-2 text-[10px] text-center text-gray-500 bg-gray-900/50 rounded-b-lg border-t border-gray-700">
        Real-time activity tracking • Auto-scrolls to latest
      </div>
    </div>
  );
};

