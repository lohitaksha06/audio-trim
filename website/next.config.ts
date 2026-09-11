import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // dev hydration/HMR breaks on 127.0.0.1 without this (Next 16 blocks
  // cross-origin dev resources by default) — allow both loopback hosts
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
