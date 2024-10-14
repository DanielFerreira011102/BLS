const META = {
	creation_year: 2024,
	title: 'BioLaySumm',
	description: 'Answering consumer health questions<br>with non-expert language',
	src: 'https://github.com/DanielFerreira011102/thesis-biolaysumm'
};

const PERSONAL = {
	name: 'Daniel Ferreira',
	first_name: 'Daniel',
	surname: 'Ferreira',
	role: 'Software Developer & Data Engineering student',
	introduction:
		'I create software that is both functional and beautiful. I have experience working with a variety of technologies, including Python, JavaScript, and SQL. I am always looking for new opportunities to learn and grow, and I am excited to see where my career will take me next.',
	location: 'Aveiro, Portugal'
};

const SKILLS = {
	specializations: [
		'Software Development',
		'Data Engineering',
		'Machine Learning',
		'Deep Learning',
		'Web Development'
	],
	programmingLanguages: ['Python', 'R', 'SQL', 'C++', 'HTML', 'CSS', 'JavaScript', 'Bash'],
	technologies: [
		'PyTorch',
		'TensorFlow',
		'Jupyter',
		'SQL Server',
		'Power BI',
		'Svelte',
		'Sveltekit',
		'React',
		'PHP',
		'Node.js',
		'MongoDB',
		'PostgreSQL',
		'Git'
	]
};

const DOCUMENTS = [
	{
		title: 'PDE_Sprint_1.pptx',
		src: '/documents/PDE_Sprint_1.pptx',
		type: 'presentation',
		description: 'This is the first sprint presentation for the PDE course.',
		download: true
	},
];


const CONTACT = {
	email: 'djbf@ua.pt',
	github: 'https://github.com/DanielFerreira011102',
	linkedin: 'https://www.linkedin.com/in/danieljbf/'
};

const NAVLIST = [
	{
		title: 'Home',
		subtitle: 'welcome',
		href: '#header',
		isCore: false
	},
	{
		title: 'Resources',
		subtitle: 'check out my work',
		href: '#resources',
		isCore: true
	},
	{
		title: 'Activities',
		subtitle: 'track my progress',
		href: '#activities',
		isCore: true
	},
	{
		title: 'About',
		subtitle: 'get to know me',
		href: '#about',
		isCore: true
	}
];

const DATA = {
	META,
	PERSONAL,
	SKILLS,
	CONTACT,
	NAVLIST,
	DOCUMENTS
};

export default DATA;
