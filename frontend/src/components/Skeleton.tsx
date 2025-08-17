import React from 'react';

const Skeleton: React.FC = () => {
  return (
    <div className="flex items-end justify-start mb-4">
      <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 animate-pulse mr-2"></div>
      <div className="w-3/4 h-12 bg-gray-300 dark:bg-gray-600 animate-pulse rounded-lg"></div>
    </div>
  );
};

export default Skeleton;
