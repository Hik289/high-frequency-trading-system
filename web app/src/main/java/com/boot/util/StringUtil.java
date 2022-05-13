package com.boot.util;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * String的工具类
 */
public class StringUtil {
	/**
	 * 是否为null 或者空字符串
	 * 
	 * @param str
	 * @return
	 */
	public static boolean isBlank(String str) {
		return str == null || str.equals("");
	}

	/**
	 * 是否不为null 或者空字符串
	 * 
	 * @param str
	 * @return
	 */
	public static boolean isNotBlank(String str) {
		return !isBlank(str);
	}

	/**
	 * find the first substr from the string whitch match the reg
	 */
	public static String find(String str, String reg) {
		if (isBlank(str)) {
			return null;
		} else {
			Pattern compile = Pattern.compile(reg);
			Matcher matcher = compile.matcher(str);
			while (matcher.find()) {
				return matcher.group(0);
			}
			return null;
		}
	}

	/**
	 * find the last substr from the string whitch match the reg
	 */
	public static String findLast(String str, String reg) {
		List<String> findAll = findAll(str, reg);
		if (findAll.size() == 0) {
			return null;
		} else {
			return findAll.get(findAll.size() - 1);
		}
	}

	/**
	 * find All substr from the string whitch match the reg
	 */
	public static List<String> findAll(String str, String reg) {
		List<String> result = new ArrayList<>();
		if (isNotBlank(str)) {
			Pattern compile = Pattern.compile(reg);
			Matcher matcher = compile.matcher(str);
			while (matcher.find()) {
				result.add(matcher.group(0));
			}
		}
		return result;
	}

	/**
	 * test the string 
	 */

	public static boolean test(String str, String reg) {
		if (str == null) {
			return false;
		}
		return Pattern.compile(reg).matcher(str).matches();
	}

	/**
	 * contact the String array to a string with split
	 */
	public static String join(String[] array, String split) {
		if (array == null || array.length == 0) {
			return null;
		} else {
			if (isBlank(split)) {
				split = "";
			}
			String result = "";
			for (String string : array) {
				result += string + split;
			}
			result = result.substring(0, result.length() - split.length());
			return result;
		}
	}

	public static String join(List<String> list, String split) {
		if (list == null || list.size() == 0) {
			return null;
		} else {
			if (isBlank(split)) {
				split = "";
			}
			String result = "";
			for (String string : list) {
				result += string + split;
			}
			result = result.substring(0, result.length() - split.length());
			return result;
		}
	}

}
