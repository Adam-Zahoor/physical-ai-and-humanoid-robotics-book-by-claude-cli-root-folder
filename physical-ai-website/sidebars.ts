import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar structure for the Physical AI and Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro', 'overview'],
    },
    {
      type: 'category',
      label: 'Module 1 - The Robotic Nervous System (ROS 2)',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Introduction to Physical AI',
          items: [
            'module-1/chapter-1/content',
            'module-1/chapter-1/applications',
            'module-1/chapter-1/glossary',
            'module-1/chapter-1/summary',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Foundations of Robotics and AI',
          items: [
            'module-1/chapter-2/content',
            'module-1/chapter-2/ai-fundamentals',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 2 - The Digital Twin (Gazebo & Unity)',
      items: [
        {
          type: 'category',
          label: 'Chapter 3: Simulation Platforms',
          items: [
            'module-2/chapter-3/content',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 4: Humanoid Robot Architectures',
          items: [
            'module-2/chapter-4/content',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 3 - The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        {
          type: 'category',
          label: 'Chapter 5: Perception and Cognition',
          items: [
            'module-3/chapter-5/content',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 6: Motion Planning and Control',
          items: [
            'module-3/chapter-6/content',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 7: Human-Robot Interaction',
          items: [
            'module-3/chapter-7/content',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 4 - Vision-Language-Action (VLA)',
      items: [
        {
          type: 'category',
          label: 'Chapter 8: AI-Native Content',
          items: [
            'module-4/chapter-8/modularization',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 9: Advanced Topics',
          items: [
            'module-4/chapter-9/reinforcement-learning',
            'module-4/chapter-9/multi-agent-systems',
            'module-4/chapter-9/implications',
            'module-4/chapter-9/future-directions',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 10: Practical Projects',
          items: [
            'module-4/chapter-10/integrated-project',
          ],
        },
      ],
    },
  ],
};

export default sidebars;
