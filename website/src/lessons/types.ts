import type { ComponentType, ReactNode } from "react";

/**
 * A lesson is a sequence of steps. Two kinds:
 *  - `explain`: teach a concept (rich JSX body), then Continue.
 *  - `mcq`: a check-for-understanding with immediate feedback.
 * Bodies are render functions so lesson content (authored in .tsx) can use CodeBlock, Callout, etc.
 */
export type Step =
  | { kind: "explain"; title?: string; body: () => ReactNode }
  | {
      kind: "mcq";
      prompt: () => ReactNode;
      options: string[];
      /** index of the correct option */
      answer: number;
      explanation: () => ReactNode;
    };

export interface Lesson {
  /** stable id, e.g. "foundations/what-is-vdj" (used for progress) */
  id: string;
  slug: string;
  track: string;
  title: string;
  summary: string;
  minutes: number;
  steps: Step[];
}

export interface Track {
  slug: string;
  title: string;
  description: string;
  icon: ComponentType<{ className?: string }>;
  /** tailwind gradient classes for the track accent */
  accent: string;
  lessons: Lesson[];
}
