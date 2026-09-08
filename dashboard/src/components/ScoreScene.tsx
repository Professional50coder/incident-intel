"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { FrameScore } from "@/lib/api";

const NORMAL_COLOR = 0x22d3ee; // cyan-400
const ANOMALY_COLOR = 0xf87171; // red-400

/**
 * Renders each frame's anomaly score as a 3D bar along a timeline - height and color both
 * encode the score, so a spike of red bars reads as "something happened here" at a glance,
 * instead of a flat line chart. Plain `three` (no react-three-fiber) to keep the dependency
 * footprint small for what is otherwise a static, non-interactive-beyond-orbit scene.
 */
export function ScoreScene({ frameScores }: { frameScores: FrameScore[] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || frameScores.length === 0) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b); // zinc-950

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    const spread = frameScores.length;
    camera.position.set(spread * 0.4, spread * 0.35 + 4, spread * 0.7 + 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.replaceChildren(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(spread / 2, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    keyLight.position.set(10, 20, 10);
    scene.add(keyLight);

    const maxScore = Math.max(...frameScores.map((f) => f.score), 1e-6);
    const barWidth = 0.6;

    frameScores.forEach((frame, i) => {
      const normalized = frame.score / maxScore;
      const barHeight = Math.max(normalized * 6, 0.05);
      const geometry = new THREE.BoxGeometry(barWidth, barHeight, barWidth);
      const material = new THREE.MeshStandardMaterial({
        color: frame.is_anomalous ? ANOMALY_COLOR : NORMAL_COLOR,
        emissive: frame.is_anomalous ? ANOMALY_COLOR : 0x000000,
        emissiveIntensity: frame.is_anomalous ? 0.4 : 0,
      });
      const bar = new THREE.Mesh(geometry, material);
      bar.position.set(i * 1.0, barHeight / 2, 0);
      scene.add(bar);
    });

    const grid = new THREE.GridHelper(spread + 4, spread + 4, 0x27272a, 0x18181b);
    grid.position.set(spread / 2 - 0.5, 0, 0);
    scene.add(grid);

    let frameId: number;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();

    function handleResize() {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      renderer.dispose();
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
    };
  }, [frameScores]);

  if (frameScores.length === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950 text-sm text-zinc-600">
        Analyze a clip to see its 3D score timeline
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-72 w-full overflow-hidden rounded-xl border border-zinc-800"
    />
  );
}
