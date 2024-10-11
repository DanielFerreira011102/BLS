import { fontFamily } from 'tailwindcss/defaultTheme';
import flowbitePlugin from 'flowbite/plugin';

/** @type {import('tailwindcss').Config} */
export default {
	content: [
		'./src/**/*.{html,js,svelte,ts}',
		'./node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}'
	],
	theme: {
		gridTemplateColumns: {
			16: 'repeat(16, minmax(0, 1fr))',
			24: 'repeat(24, minmax(0, 1fr))'
		},
		screens: {
			xs: '520px',
			sm: '640px',
			md: '768px',
			lg: '1024px',
			xl: '1280px'
		},
		colors: {
			'light-gray': '#f1f1f1'
		},
		extend: {
			lineHeight: {
				11: '2.75rem',
				12: '3rem',
				14: '3.5rem',
				16: '4rem'
			},
			fontFamily: {
				sans: ['Saans', ...fontFamily.sans],
				serif: ['Domine Variable', ...fontFamily.serif],
				mono: ['JetBrains Mono Variable', ...fontFamily.mono]
			}
		}
	},
	plugins: [flowbitePlugin]
};
