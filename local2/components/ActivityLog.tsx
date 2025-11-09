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
          <svg className="h-3.5 w-3.5 text-green-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'error':
        return (
          <svg className="h-3.5 w-3.5 text-red-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="h-3.5 w-3.5 text-yellow-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        );
      default:
        return (
          <svg className="h-3.5 w-3.5 text-blue-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
    <div className="bg-gray-800 rounded-lg shadow-xl border border-blue-500 h-full flex flex-col">
      <div className="flex-shrink-0 flex items-center justify-between p-2 bg-gray-700/50 rounded-t-lg border-b border-gray-600">
        <h2 className="text-sm font-semibold text-blue-400 flex items-center gap-1.5">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span>Activity</span>
          {activities.length > 0 && (
            <span className="ml-1 px-1.5 py-0.5 bg-blue-500/30 text-blue-300 text-[10px] rounded-full">
              {activities.length}
            </span>
          )}
        </h2>
        {activities.length > 0 && (
          <button
            onClick={onClear}
            className="px-2 py-0.5 text-[10px] bg-gray-600 hover:bg-gray-500 text-gray-300 rounded transition-colors"
          >
            Clear
          </button>
        )}
      </div>
      <div className="flex-1 min-h-0 max-h-[600px] p-2 font-mono text-[10px] text-gray-300 overflow-y-auto bg-gray-900/50">
        {activities.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center px-2">
              <svg className="mx-auto h-6 w-6 mb-1 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-[10px]">No activity</p>
            </div>
          </div>
        ) : (
          <div className="space-y-1.5">
            {activities.map((activity) => (
              <div
                key={activity.id}
                className="flex items-start gap-2 p-1.5 rounded bg-gray-800/50 hover:bg-gray-800 transition-colors"
              >
                <div className="flex-shrink-0 mt-0.5">{getIcon(activity.type)}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-gray-500 text-[9px] font-mono">
                      {formatTime(activity.timestamp)}
                    </span>
                    <span className={`font-semibold text-[10px] ${getTextColor(activity.type)}`}>
                      {activity.action}
                    </span>
                  </div>
                  <p className="text-gray-400 text-[10px] break-words mt-0.5 leading-tight">{activity.message}</p>
                </div>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        )}
      </div>
      <div className="flex-shrink-0 p-1.5 text-[9px] text-center text-gray-500 bg-gray-900/50 rounded-b-lg border-t border-gray-700">
        Real-time • Auto-scroll
      </div>
    </div>
  );
};

