import type { Metadata } from 'next';
import React from 'react';

export const metadata: Metadata = {
  title: 'TurfVision - 8v8 Turf AI Analytics',
  description: '8v8 football match analytics and 3D goal reconstruction platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
