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
		'I\'m a Master\'s student in Software Engineering and Computer Science at the University of Aveiro, specializing in artificial intelligence and natural language processing. My research focuses on machine learning and large language models, with particular emphasis on deep learning and transformer architectures.\nCurrently, my thesis work involves developing language models for biomedical question-answering with controllable generation, combining my passion for AI advancement with practical healthcare applications',
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
