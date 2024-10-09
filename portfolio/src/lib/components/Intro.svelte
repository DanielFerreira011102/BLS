<script>
    import { Icon } from 'svelte-icons-pack';
    import { FaSolidArrowRight } from "svelte-icons-pack/fa";

    export let personal;
</script>

<section class="container mx-auto font-sans" id="intro">
    <h1 class="text-4xl font-semibold text-black m-0">Hi, I'm {personal.first_name}</h1>
    <span class="text-2xl text-gray-500">{personal.role} from {personal.location}</span>
    <p class="text-xl py-2 text-dark-grayish-blue">{personal.introduction}</p>
    <div class="font-mono font-bold mt-4">
        <a href="/about" class="bg-black text-white py-2 px-4 hover:bg-gray-900 inline-flex items-center">
            More Information
            <Icon src={FaSolidArrowRight} className="inline ml-2" />
        </a>
    </div>
    <div class="border-t border-gray-300 my-9"></div>
</section>
