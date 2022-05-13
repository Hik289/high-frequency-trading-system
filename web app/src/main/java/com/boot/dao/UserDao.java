package com.boot.dao;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface UserDao {
	
	List<?> feesEchart();
	List<?> feesEchartTimeUser();
	
	List<?> myFeesEchart(@Param("userId")Integer userId);
}
