<script>
    export let personal;
</script>

<section class="container mx-auto font-sans" id="intro">
    <h1 class="text-4xl font-semibold text-black m-0">Hi, I'm {personal.first_name}</h1>
    <span class="text-2xl text-gray-500">{personal.role} from {personal.location}</span>
    <p class="text-xl py-2 text-dark-grayish-blue">{personal.introduction}</p>
    <div class="font-mono font-bold mt-4">
        <a href="/about" class="bg-gray-900 text-white py-2 px-4 rounded hover:bg-gray-800">More Information →</a>
    </div>
    <div class="border-t border-gray-300 my-9"></div>
</section>