import { fontFamily } from 'tailwindcss/defaultTheme';
import flowbitePlugin from 'flowbite/plugin'

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}', './node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      lineHeight: {
        '11': '2.75rem',
        '12': '3rem',
      },
      fontFamily: {
        sans: [
          'Saans',
          ...fontFamily.sans
        ],
        mono: [
          'JetBrains Mono Variable',
          ...fontFamily.mono
        ],
      },
    },
  },
  plugins: [flowbitePlugin],
}

